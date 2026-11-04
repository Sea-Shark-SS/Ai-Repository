"""
src/lc_qa.py
The same question answering as qa.py, built with LangChain.

  * PaperRetriever   - a LangChain BaseRetriever that wraps our ChromaDB/FAISS
                       Retriever and returns LangChain Documents
  * get_llm()        - the chat model (ChatAnthropic, ChatOpenAI or ChatOllama),
                       chosen with the same LLM_PROVIDER / LLM_MODEL variables as qa.py
  * the chain        - prompt | chat model | StrOutputParser  (LangChain Expression Language)

    from src.retriever import Retriever
    from src.lc_qa import answer_question_langchain

    result = answer_question_langchain("What is IVF?", Retriever())

Install:   pip install langchain-core
plus ONE of: langchain-anthropic | langchain-openai | langchain-ollama
"""

import os
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from src.qa import (
    DEFAULT_MODELS,
    MAX_ANSWER_TOKENS,
    MIN_SCORE,
    NOT_FOUND_MESSAGE,
    OLLAMA_URL,
    SYSTEM_PROMPT,
    TIMEOUT_SECONDS,
    TOP_K,
    LLMError,
    build_context,
    pick_provider,
)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class PaperRetriever(BaseRetriever):
    """LangChain retriever on top of src.retriever.Retriever."""

    paper_store: Any                 # a src.retriever.Retriever
    top_k: int = TOP_K
    source: Any = None               # one PDF name, a list of names, or None for all
    min_score: float = MIN_SCORE

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        hits = self.paper_store.search(query, top_k=self.top_k, source=self.source)
        docs = []
        for h in hits:
            if h["score"] < self.min_score:
                continue
            metadata = {k: v for k, v in h.items() if k != "text"}
            docs.append(Document(page_content=h["text"], metadata=metadata))
        return docs


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------
def get_llm():
    """Create the LangChain chat model for the provider chosen in the environment."""
    provider = pick_provider()
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise LLMError("Missing package. Run: pip install langchain-anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(model=model, max_tokens=MAX_ANSWER_TOKENS, timeout=TIMEOUT_SECONDS)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise LLMError("Missing package. Run: pip install langchain-openai")
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=model, max_tokens=MAX_ANSWER_TOKENS, timeout=TIMEOUT_SECONDS)

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise LLMError("Missing package. Run: pip install langchain-ollama")
    return ChatOllama(model=model, base_url=OLLAMA_URL, num_predict=MAX_ANSWER_TOKENS)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------
def build_chain(llm):
    """prompt | chat model | string parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context passages:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    return prompt | llm | StrOutputParser()


def answer_question_langchain(question, retriever, top_k=TOP_K, source=None, llm=None):
    """
    Same result format as qa.answer_question:

        {"answer": str, "sources": [{"source", "page", "title", "author", "score", "text"}, ...]}
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("The question is empty.")

    docs = PaperRetriever(paper_store=retriever, top_k=top_k, source=source).invoke(question)
    if not docs:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    results = [{**d.metadata, "text": d.page_content} for d in docs]
    chain = build_chain(llm or get_llm())
    try:
        answer = chain.invoke({"context": build_context(results), "question": question})
    except Exception as e:  # network, auth, bad model name, rate limit ...
        raise LLMError(f"The language model request failed: {e}")

    answer = (answer or "").strip()
    if not answer:
        raise LLMError("The language model returned an empty answer.")
    return {"answer": answer, "sources": results}
