"""
pdf_embedder.py
Read a PDF, split it into chunks, turn each chunk into an embedding vector,
and search the chunks by meaning. This is the retrieval half of a RAG pipeline.

Install:
    pip install pymupdf sentence-transformers numpy

Usage:
    python pdf_embedder.py index paper.pdf                  # build and save the index
    python pdf_embedder.py search paper.pdf "your question" # search the saved index

Every TODO marks a choice you should review or customize.
"""

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# Embedding model: 12-layer MiniLM, 384 dimensions. Slightly more accurate than
# the 6-layer version (all-MiniLM-L6-v2) but a bit slower.
# Note: it truncates input at 128 tokens by default (roughly 500-600
# characters), so chunks are kept at 500 characters to avoid losing text.
MODEL_NAME = "all-MiniLM-L12-v2"

# TODO: tune chunk size (characters) and overlap. Smaller chunks give more
# precise matches, larger chunks keep more context. If you raise CHUNK_SIZE
# above ~550, also raise the model's limit: model.max_seq_length = 256
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# TODO: how many chunks to return for a question.
TOP_K = 4

INDEX_DIR = Path("index")  # TODO: change where indexes are stored if you like


# ---------------------------------------------------------------------------
# 1. Read the PDF
# ---------------------------------------------------------------------------
def read_pdf(path):
    """Return a list of (page_number, text) for every page that has text."""
    pages = []
    with fitz.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append((number, text))
    # TODO: scanned PDFs have no text layer, so this returns nothing.
    # Add OCR (for example pytesseract) when `pages` is empty.
    if not pages:
        raise ValueError(f"No extractable text found in {path}. It may be a scanned PDF.")
    return pages


# ---------------------------------------------------------------------------
# 2. Split into chunks
# ---------------------------------------------------------------------------
def chunk_pages(pages, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split each page into overlapping chunks and remember the page number."""
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
    chunks = []
    for page_number, text in pages:
        text = " ".join(text.split())  # collapse newlines and extra spaces
        # TODO: splitting by characters can cut sentences in half. Split on
        # sentences or paragraphs instead for cleaner chunks.
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": page_number, "text": piece})
            start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. Embed the chunks
# ---------------------------------------------------------------------------
def embed_texts(texts, model):
    """Turn a list of strings into a (len(texts), dim) array of unit vectors."""
    vectors = model.encode(
        texts,
        batch_size=32,  # TODO: lower this if you run out of memory
        show_progress_bar=True,
        normalize_embeddings=True,  # unit length, so dot product = cosine similarity
    )
    return np.asarray(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# 4. Save and load the index
# ---------------------------------------------------------------------------
def index_paths(pdf_path):
    stem = Path(pdf_path).stem
    return INDEX_DIR / f"{stem}.npy", INDEX_DIR / f"{stem}.json"


def build_index(pdf_path):
    print(f"Reading {pdf_path} ...")
    pages = read_pdf(pdf_path)
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    vectors = embed_texts([c["text"] for c in chunks], model)

    INDEX_DIR.mkdir(exist_ok=True)
    vec_path, meta_path = index_paths(pdf_path)
    np.save(vec_path, vectors)
    meta_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {vec_path} and {meta_path}")
    # TODO: for big documents or many PDFs, store vectors in FAISS or Chroma
    # instead of a .npy file.


def load_index(pdf_path):
    vec_path, meta_path = index_paths(pdf_path)
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No index for {pdf_path}. Run: python pdf_embedder.py index {pdf_path}")
    vectors = np.load(vec_path)
    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    return vectors, chunks


# ---------------------------------------------------------------------------
# 5. Search
# ---------------------------------------------------------------------------
def search(pdf_path, question, top_k=TOP_K):
    vectors, chunks = load_index(pdf_path)
    model = SentenceTransformer(MODEL_NAME)  # must be the same model used to build the index
    query = embed_texts([question], model)[0]

    scores = vectors @ query  # cosine similarity because all vectors are normalized
    best = np.argsort(scores)[::-1][:top_k]
    return [{"score": float(scores[i]), **chunks[i]} for i in best]


# TODO: this is where the "generation" half of RAG goes. Send `question` plus
# the retrieved chunks to your LLM and return its answer, for example:
#
# def answer(question, results):
#     context = "\n\n".join(f"[page {r['page']}] {r['text']}" for r in results)
#     prompt = f"Answer using only this context.\n\n{context}\n\nQuestion: {question}"
#     return call_your_llm(prompt)   # TODO: implement with your LLM of choice


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) >= 3 and argv[1] == "index":
        build_index(argv[2])
    elif len(argv) >= 4 and argv[1] == "search":
        for r in search(argv[2], " ".join(argv[3:])):
            print(f"\n[page {r['page']}] score={r['score']:.3f}")
            print(r["text"])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
