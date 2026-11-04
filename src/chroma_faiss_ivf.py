"""
chroma_faiss_ivf.py
Vector search over PDFs using two popular tools together:

  * ChromaDB  - persistent store for chunk text, metadata and embeddings
  * FAISS IVF - fast approximate nearest-neighbour search (IndexIVFFlat)

Why both? ChromaDB's own index is HNSW and cannot be switched to IVF, so Chroma
is the source of truth and FAISS builds an IVF index from the vectors stored in it.

Needs pdf_embedder.py in the same folder (it provides PDF reading, chunking and
the embedding model).

Install:
    pip install chromadb faiss-cpu pymupdf sentence-transformers numpy

Usage:
    python chroma_faiss_ivf.py ingest paper.pdf      # read, chunk, embed, store in Chroma
    python chroma_faiss_ivf.py build                 # build the FAISS IVF index from Chroma
    python chroma_faiss_ivf.py search "your question"
    python chroma_faiss_ivf.py stats

Run `build` again after ingesting new PDFs, because IVF is trained on a fixed set of vectors.
"""

import json
import sys
from pathlib import Path

import chromadb
import faiss
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Reuse the PDF reader, chunker and embedding function from the first script.
from pdf_embedder import MODEL_NAME, chunk_pages, embed_texts, read_pdf

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION = "pdf_chunks"          # TODO: use a different name per project
FAISS_FILE = Path("chroma_db/ivf.faiss")
IDS_FILE = Path("chroma_db/ivf_ids.json")

# TODO: IVF tuning.
#   nlist  = number of clusters. Rule of thumb: about sqrt(N) for small data,
#            4*sqrt(N) up to 16*sqrt(N) for large data (N = number of chunks).
#   nprobe = clusters searched per query. Higher = more accurate but slower.
#            nprobe == nlist is exact search.
NLIST_FACTOR = 1.0
NPROBE = 8

TOP_K = 4                          # TODO: number of chunks to return
BATCH = 1000                       # Chroma limits how many items one add() call accepts


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    # Our vectors are normalized, so cosine and inner product agree.
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(pdf_path):
    """Read a PDF, embed its chunks and store everything in ChromaDB."""
    print(f"Reading {pdf_path} ...")
    chunks = chunk_pages(read_pdf(pdf_path))
    print(f"{len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    vectors = embed_texts([c["text"] for c in chunks], model)

    stem = Path(pdf_path).stem
    ids = [f"{stem}-{i}" for i in range(len(chunks))]
    metas = [{"source": Path(pdf_path).name, "page": c["page"]} for c in chunks]
    docs = [c["text"] for c in chunks]

    col = get_collection()
    for s in range(0, len(ids), BATCH):
        e = s + BATCH
        # upsert, so ingesting the same PDF twice replaces instead of duplicating
        col.upsert(ids=ids[s:e], embeddings=vectors[s:e].tolist(), documents=docs[s:e], metadatas=metas[s:e])
    print(f"Stored. Collection now holds {col.count()} chunks. Run `build` to refresh the IVF index.")
    # TODO: to remove a PDF later, use col.delete(where={"source": "paper.pdf"})


# ---------------------------------------------------------------------------
# FAISS IVF
# ---------------------------------------------------------------------------
def build_ivf():
    """Pull all vectors out of ChromaDB and train a FAISS IVF index on them."""
    col = get_collection()
    n = col.count()
    if n == 0:
        raise SystemExit("ChromaDB is empty. Run `ingest <pdf>` first.")

    data = col.get(include=["embeddings"], limit=n)
    ids = list(data["ids"])
    vectors = np.ascontiguousarray(np.asarray(data["embeddings"], dtype="float32"))
    dim = vectors.shape[1]

    nlist = max(1, min(n, int(NLIST_FACTOR * np.sqrt(n))))
    quantizer = faiss.IndexFlatIP(dim)  # coarse quantizer: assigns vectors to clusters
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    print(f"Training IVF on {n} vectors (dim={dim}, nlist={nlist}) ...")
    # FAISS suggests at least ~39 training points per cluster; with fewer it prints a warning.
    index.train(vectors)
    index.add(vectors)

    FAISS_FILE.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_FILE))
    IDS_FILE.write_text(json.dumps(ids), encoding="utf-8")
    print(f"Saved {FAISS_FILE}")
    # TODO: for millions of vectors, consider IndexIVFPQ (compressed vectors)
    # to cut memory, at a small cost in accuracy.


def load_ivf():
    if not FAISS_FILE.exists() or not IDS_FILE.exists():
        raise SystemExit("No IVF index found. Run `build` first.")
    index = faiss.read_index(str(FAISS_FILE))
    ids = json.loads(IDS_FILE.read_text(encoding="utf-8"))
    return index, ids


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search(question, top_k=TOP_K):
    index, ids = load_ivf()
    index.nprobe = min(NPROBE, index.nlist)

    model = SentenceTransformer(MODEL_NAME)  # must match the model used at ingest time
    query = embed_texts([question], model)

    scores, positions = index.search(query, top_k)
    hits = [(int(p), float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]
    if not hits:
        return []

    # Fetch the text and metadata for the winning ids from ChromaDB.
    wanted = [ids[p] for p, _ in hits]
    got = get_collection().get(ids=wanted, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}

    results = []
    for (p, score) in hits:
        doc, meta = by_id[ids[p]]
        results.append({"score": score, "text": doc, **meta})
    return results
    # TODO: add a metadata filter (for example only one PDF) by searching a
    # larger top_k and filtering, or by building one IVF index per source.


def stats():
    col = get_collection()
    print(f"Chroma collection '{COLLECTION}': {col.count()} chunks")
    if FAISS_FILE.exists():
        index, _ = load_ivf()
        print(f"FAISS IVF index: {index.ntotal} vectors, nlist={index.nlist}, dim={index.d}")
    else:
        print("FAISS IVF index: not built yet")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "ingest" and len(argv) >= 3:
        ingest(argv[2])
    elif cmd == "build":
        build_ivf()
    elif cmd == "search" and len(argv) >= 3:
        for r in search(" ".join(argv[2:])):
            print(f"\n[{r['source']} p.{r['page']}] score={r['score']:.3f}")
            print(r["text"])
    elif cmd == "stats":
        stats()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
