"""
src/retriever.py
Vector search over PDF chunks.

  * ChromaDB   - source of truth: stores chunk text, metadata and embeddings
  * FAISS IVF  - fast approximate search, built from the vectors in Chroma

search() uses the FAISS IVF index when it exists and is up to date, and falls
back to ChromaDB's own search otherwise, so it always works even if you forget
to run build_ivf() after ingesting a new PDF.

Usage from other files (for example app.py or qa.py):

    from src.retriever import Retriever

    r = Retriever()
    r.ingest("data/paper.pdf")
    r.build_ivf()
    results = r.search("What is the main contribution?")
    # each result: {"score": float, "text": str, "source": str, "page": int}
"""

import json
import sys
from pathlib import Path

import chromadb
import faiss
import numpy as np
from chromadb.config import Settings

from src.chunking import chunk_pages
from src.embeddings import embed_texts, load_model
from src.loader import read_pdf

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION = "pdf_chunks"   # TODO: use a different name per project

# IVF tuning
#   nlist  = number of clusters, about sqrt(N) for small data (N = chunks)
#   nprobe = clusters searched per query. Higher = more accurate but slower.
#            nprobe == nlist is exact search.
NLIST_FACTOR = 1.0
NPROBE = 8

TOP_K = 4
BATCH = 1000                # Chroma limits how many items one upsert accepts


class Retriever:
    def __init__(self, chroma_dir=CHROMA_DIR, collection=COLLECTION):
        chroma_dir = Path(chroma_dir)
        self.faiss_file = chroma_dir / "ivf.faiss"
        self.ids_file = chroma_dir / "ivf_ids.json"

        client = chromadb.PersistentClient(
            path=str(chroma_dir), settings=Settings(anonymized_telemetry=False)
        )
        # Vectors are normalized, so cosine and inner product agree.
        self.col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

        self._model = None   # loaded once, on first use
        self._ivf = None
        self._ivf_ids = None

    # ----- embedding model ------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            self._model = load_model()
        return self._model

    # ----- ingest ---------------------------------------------------------
    def ingest(self, pdf_path):
        """Read a PDF, chunk it, embed the chunks and store them in ChromaDB."""
        pdf_path = Path(pdf_path)
        chunks = chunk_pages(read_pdf(pdf_path))
        vectors = embed_texts([c["text"] for c in chunks], self.model)

        ids = [f"{pdf_path.stem}-{i}" for i in range(len(chunks))]
        docs = [c["text"] for c in chunks]
        metas = [{"source": pdf_path.name, "page": c["page"]} for c in chunks]

        for s in range(0, len(ids), BATCH):
            e = s + BATCH
            # upsert, so ingesting the same PDF twice replaces instead of duplicating
            self.col.upsert(
                ids=ids[s:e],
                embeddings=vectors[s:e].tolist(),
                documents=docs[s:e],
                metadatas=metas[s:e],
            )
        self._ivf = None  # the IVF index is now out of date
        return len(chunks)

    def delete_pdf(self, pdf_name):
        """Remove every chunk that came from one PDF (pass the file name)."""
        self.col.delete(where={"source": pdf_name})
        self._ivf = None

    # ----- FAISS IVF ------------------------------------------------------
    def build_ivf(self):
        """Pull all vectors out of ChromaDB and train a FAISS IVF index on them."""
        n = self.col.count()
        if n == 0:
            raise ValueError("ChromaDB is empty. Ingest a PDF first.")

        data = self.col.get(include=["embeddings"], limit=n)
        ids = list(data["ids"])
        vectors = np.ascontiguousarray(np.asarray(data["embeddings"], dtype="float32"))
        dim = vectors.shape[1]

        nlist = max(1, min(n, int(NLIST_FACTOR * np.sqrt(n))))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)

        self.faiss_file.parent.mkdir(exist_ok=True)
        faiss.write_index(index, str(self.faiss_file))
        self.ids_file.write_text(json.dumps(ids), encoding="utf-8")

        self._ivf, self._ivf_ids = index, ids
        return n, nlist

    def _ivf_ready(self):
        """True if a saved IVF index exists and covers every chunk in Chroma."""
        if self._ivf is None:
            if not (self.faiss_file.exists() and self.ids_file.exists()):
                return False
            self._ivf = faiss.read_index(str(self.faiss_file))
            self._ivf_ids = json.loads(self.ids_file.read_text(encoding="utf-8"))
        if self._ivf.ntotal != self.col.count():
            self._ivf = None  # stale: PDFs were added or removed since the build
            return False
        self._ivf.nprobe = min(NPROBE, self._ivf.nlist)
        return True

    # ----- search ---------------------------------------------------------
    def search(self, question, top_k=TOP_K, source=None):
        """Return the top_k most relevant chunks. Optionally limit to one PDF name."""
        if self.col.count() == 0:
            return []
        query = embed_texts([question], self.model)  # shape (1, dim)

        if self._ivf_ready():
            return self._search_ivf(query, top_k, source)
        return self._search_chroma(query, top_k, source)

    def _search_ivf(self, query, top_k, source):
        # With a source filter, fetch extra candidates and filter afterwards.
        k = min(top_k * 10 if source else top_k, self._ivf.ntotal)
        scores, positions = self._ivf.search(query, k)
        hits = [(self._ivf_ids[p], float(s)) for p, s in zip(positions[0], scores[0]) if p != -1]
        if not hits:
            return []

        got = self.col.get(ids=[i for i, _ in hits], include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}

        results = []
        for chunk_id, score in hits:
            doc, meta = by_id[chunk_id]
            if source and meta["source"] != source:
                continue
            results.append({"score": score, "text": doc, **meta})
        return results[:top_k]

    def _search_chroma(self, query, top_k, source):
        res = self.col.query(
            query_embeddings=query.tolist(),
            n_results=min(top_k, self.col.count()),
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            # Chroma cosine distance = 1 - cosine similarity
            results.append({"score": 1.0 - float(dist), "text": doc, **meta})
        return results

    # ----- info -----------------------------------------------------------
    def stats(self):
        info = {"chunks": self.col.count(), "ivf_built": self._ivf_ready()}
        if info["ivf_built"]:
            info["nlist"] = self._ivf.nlist
        return info


# ---------------------------------------------------------------------------
# Quick test from the command line (run from the project root):
#   python -m src.retriever ingest data/paper.pdf
#   python -m src.retriever build
#   python -m src.retriever search "your question"
#   python -m src.retriever stats
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    r = Retriever()
    if cmd == "ingest" and len(sys.argv) >= 3:
        print(f"Stored {r.ingest(sys.argv[2])} chunks. Total: {r.col.count()}")
    elif cmd == "build":
        n, nlist = r.build_ivf()
        print(f"Built IVF index on {n} vectors (nlist={nlist})")
    elif cmd == "search" and len(sys.argv) >= 3:
        for hit in r.search(" ".join(sys.argv[2:])):
            print(f"\n[{hit['source']} p.{hit['page']}] score={hit['score']:.3f}")
            print(hit["text"])
    elif cmd == "stats":
        print(r.stats())
    else:
        print(__doc__)
