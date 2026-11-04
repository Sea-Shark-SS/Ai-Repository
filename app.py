"""
app.py - Research Paper RAG Assistant (Streamlit)

Upload research papers (PDF), then ask questions and get answers with citations.

Run from the project root:
    streamlit run app.py
"""

import hashlib
import os
from pathlib import Path

import streamlit as st

from src.qa import DEFAULT_MODELS, LLMError, answer_question, format_citation, pick_provider
from src.retriever import Retriever, safe_filename

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

st.set_page_config(page_title="Research Paper RAG Assistant", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Opening the paper library ...")
def get_retriever():
    return Retriever(chroma_dir=CHROMA_DIR)


st.session_state.setdefault("messages", [])      # chat history
st.session_state.setdefault("uploader_key", 0)   # changing it resets the upload box
st.session_state.setdefault("processed", {})     # file name -> hash of the version already indexed
st.session_state.setdefault("failed", {})        # file name -> (hash, error message)

try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"Could not open the paper database: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def handle_uploads(files):
    """Index every newly uploaded PDF. Files that were already indexed are skipped."""
    current_names = {safe_filename(f.name) for f in files}
    for name in list(st.session_state.failed):
        if name not in current_names:          # the file was removed from the upload box
            del st.session_state.failed[name]

    known = {p["source"] for p in retriever.list_papers()}
    added = False
    for f in files:
        name = safe_filename(f.name)
        data = f.getvalue()
        digest = hashlib.md5(data).hexdigest()

        if name in known:
            if st.session_state.processed.get(name) == digest:
                continue
            saved = UPLOAD_DIR / name          # same file uploaded in an earlier session?
            if saved.exists() and hashlib.md5(saved.read_bytes()).hexdigest() == digest:
                st.session_state.processed[name] = digest
                continue
        failure = st.session_state.failed.get(name)
        if failure and failure[0] == digest:   # failed before: show the message, do not retry
            st.error(failure[1])
            continue

        try:
            with st.spinner(f"Reading and indexing {f.name} (the first time also loads the embedding model) ..."):
                source, n_chunks = retriever.ingest_bytes(f.name, data, UPLOAD_DIR)
        except Exception as e:
            message = f"❌ {f.name}: {e}"
            st.session_state.failed[name] = (digest, message)
            st.error(message)
            continue

        st.session_state.processed[source] = digest
        st.session_state.failed.pop(name, None)
        added = True
        st.success(f"✅ {source}: {n_chunks} passages indexed")

    if added:
        with st.spinner("Updating the search index ..."):
            retriever.refresh_index()


def run_qa(question, scope, top_k, engine):
    """Call the chosen pipeline. Returns {"answer": ..., "sources": [...]}."""
    if engine == "LangChain":
        from src.lc_qa import answer_question_langchain  # needs langchain-core
        return answer_question_langchain(question, retriever, top_k=top_k, source=scope)
    return answer_question(question, retriever, top_k=top_k, source=scope)


def render_sources(sources):
    if not sources:
        return
    st.markdown("**Sources**")
    for number, s in enumerate(sources, start=1):
        with st.expander(f"[{number}] {format_citation(s)}  ·  relevance {s['score']:.2f}"):
            st.markdown(s["text"].replace("$", r"\$"))


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"))


# ---------------------------------------------------------------------------
# Sidebar: upload, library, settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Paper library")

    uploaded = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded:
        handle_uploads(uploaded)

    papers = retriever.list_papers()
    sources = [p["source"] for p in papers]

    if papers:
        for p in papers:
            label = p["title"] or p["source"]
            details = f"{p['source']} · {p['pages']} pages · {p['chunks']} passages"
            if p["author"]:
                details = f"{p['author']} · {details}"
            st.markdown(f"**{label}**  \n<small>{details}</small>", unsafe_allow_html=True)

        to_remove = st.selectbox(
            "Remove a paper", options=[""] + sources, format_func=lambda s: s or "Choose a paper ..."
        )
        if st.button("Remove selected paper", disabled=not to_remove):
            retriever.delete_pdf(to_remove)
            (UPLOAD_DIR / Path(to_remove).name).unlink(missing_ok=True)
            st.session_state.processed.pop(to_remove, None)
            st.session_state.uploader_key += 1   # reset the upload box so the file is not re-added
            retriever.refresh_index()
            st.rerun()
    else:
        st.caption("No papers yet. Upload a PDF to begin.")

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Passages to retrieve", min_value=2, max_value=10, value=4)
    engine = st.radio(
        "Pipeline",
        ["From scratch", "LangChain"],
        help="Both use the same ChromaDB + FAISS retriever. 'LangChain' runs the "
             "question answering through a LangChain chain (needs langchain-core).",
    )
    selected = st.multiselect("Search in", options=sources, default=sources)

    try:
        provider = pick_provider()
        model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
        st.caption(f"Language model: {provider} · {model}")
    except LLMError as e:
        st.caption(f"⚠️ {e}")

    stats = retriever.stats()
    if stats["ivf_built"]:
        st.caption(f"Search index: FAISS IVF ({stats['chunks']} passages, nlist={stats['nlist']})")
    else:
        st.caption(f"Search index: ChromaDB ({stats['chunks']} passages)")
    if st.button("Build FAISS IVF index now", disabled=stats["chunks"] == 0):
        with st.spinner("Training the IVF index ..."):
            n, nlist = retriever.build_ivf()
        st.success(f"IVF index built on {n} passages (nlist={nlist})")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area: chat
# ---------------------------------------------------------------------------
st.title("Research Paper RAG Assistant")
st.caption("Upload papers in the sidebar, then ask questions. Answers are built only from the papers and cite their sources.")

if not papers:
    st.info("👈 Upload one or more research papers (PDF) to get started.")
elif not selected:
    st.warning("Select at least one paper under **Search in** in the sidebar.")

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask a question about your papers ...", disabled=not (papers and selected))

if question:
    question = question.strip()
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    scope = None if set(selected) == set(sources) else selected   # None = search every paper
    sources_found = []
    with st.chat_message("assistant"):
        with st.spinner("Searching the papers and writing an answer ..."):
            try:
                result = run_qa(question, scope, top_k, engine)
                content, sources_found = result["answer"], result["sources"]
            except LLMError as e:
                # The language model is not available, but retrieval still works.
                content = f"⚠️ {e}"
                try:
                    sources_found = retriever.search(question, top_k=top_k, source=scope)
                    if sources_found:
                        content += "\n\nHere are the most relevant passages I found:"
                except Exception:
                    sources_found = []
            except ImportError as e:
                content = f"⚠️ A required package is missing: {e}. Install it with pip, or switch the pipeline to 'From scratch'."
            except ValueError as e:
                content = f"⚠️ {e}"
            except Exception as e:
                content = f"⚠️ Unexpected error: {e}"

        assistant_message = {"role": "assistant", "content": content, "sources": sources_found}
        st.markdown(content)
        render_sources(sources_found)
    st.session_state.messages.append(assistant_message)
