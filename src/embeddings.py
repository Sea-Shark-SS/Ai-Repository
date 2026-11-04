"""
src/embeddings.py
The embedding model: turns text into unit-length vectors.

The same model must be used for the PDF chunks and for the questions.
If you change MODEL_NAME, delete the chroma_db/ folder and ingest again.
"""

import numpy as np

# 12-layer MiniLM, 384 dimensions. Reads at most 128 tokens (about 500-600
# characters) per text, which is why chunks are kept at 500 characters.
MODEL_NAME = "all-MiniLM-L12-v2"


def load_model(model_name=MODEL_NAME):
    """Load the sentence-transformers model (downloads it the first time)."""
    # Imported here so the rest of the project can be imported without
    # loading PyTorch until a model is really needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts, model):
    """Turn a list of strings into a (len(texts), dim) float32 array of unit vectors."""
    texts = list(texts)
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype="float32")

    vectors = model.encode(
        texts,
        batch_size=32,                      # lower this if you run out of memory
        show_progress_bar=len(texts) > 1,   # no progress bar for a single question
        normalize_embeddings=True,          # unit length, so dot product = cosine similarity
    )
    return np.asarray(vectors, dtype="float32")
