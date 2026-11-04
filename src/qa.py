"""
src/qa.py
Prompt building and answer generation (the "G" in RAG).

    from src.retriever import Retriever
    from src.qa import answer_question

    result = answer_question("What is IVF?", Retriever())
    print(result["answer"])
    for s in result["sources"]:
        print(s["source"], s["page"])

Choosing the language model (set environment variables, no code changes):

    LLM_PROVIDER   anthropic | openai | ollama
                   (if not set: anthropic when ANTHROPIC_API_KEY exists,
                    else openai when OPENAI_API_KEY exists, else ollama)
    LLM_MODEL      model name (each provider has a default, see DEFAULT_MODELS)

    anthropic: pip install anthropic   and set ANTHROPIC_API_KEY
    openai:    pip install openai      and set OPENAI_API_KEY
    ollama:    free and local. Install Ollama, run `ollama pull llama3.2`,
               no pip package and no key needed.
"""

import json
import os
import sys
import urllib.error
import urllib.request

TOP_K = 4
MIN_SCORE = 0.05          # chunks scoring below this are treated as unrelated (raise to be stricter)
MAX_ANSWER_TOKENS = 700
TIMEOUT_SECONDS = 120

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

NOT_FOUND_MESSAGE = "I could not find anything about that in the PDF."

SYSTEM_PROMPT = (
    "You answer questions about a PDF document. "
    "Use ONLY the numbered context passages you are given. "
    "If the passages do not contain the answer, say that the document does not "
    "contain it - do not guess and do not use outside knowledge. "
    "After each statement, cite the passage it came from like [1] or [2]. "
    "Keep the answer clear and concise."
)


class LLMError(Exception):
    """Raised when the language model cannot be called or returns an error."""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_context(results):
    """Format retrieved chunks as numbered passages with their source and page."""
    blocks = []
    for number, r in enumerate(results, start=1):
        blocks.append(f"[{number}] ({r['source']}, page {r['page']})\n{r['text']}")
    return "\n\n".join(blocks)


def build_prompt(question, results):
    """Return the user message: the context passages followed by the question."""
    return f"Context passages:\n\n{build_context(results)}\n\nQuestion: {question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
def pick_provider():
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider:
        if provider not in DEFAULT_MODELS:
            raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(DEFAULT_MODELS)}.")
        return provider
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _call_anthropic(model, system, user):
    try:
        import anthropic
    except ImportError:
        raise LLMError("The 'anthropic' package is missing. Run: pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError("ANTHROPIC_API_KEY is not set.")
    try:
        client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS)
        response = client.messages.create(
            model=model,
            max_tokens=MAX_ANSWER_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:  # network, auth, bad model name, rate limit ...
        raise LLMError(f"Anthropic request failed: {e}")
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    return text.strip()


def _call_openai(model, system, user):
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("The 'openai' package is missing. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise LLMError("OPENAI_API_KEY is not set.")
    try:
        client = OpenAI(timeout=TIMEOUT_SECONDS)
        response = client.chat.completions.create(
            model=model,
            max_tokens=MAX_ANSWER_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    except Exception as e:
        raise LLMError(f"OpenAI request failed: {e}")
    return (response.choices[0].message.content or "").strip()


def _call_ollama(model, system, user):
    payload = json.dumps({
        "model": model,
        "stream": False,
        "options": {"num_predict": MAX_ANSWER_TOKENS},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"Ollama returned an error ({e.code}). Did you run `ollama pull {model}`?")
    except (urllib.error.URLError, OSError):
        raise LLMError(f"Cannot reach Ollama at {OLLAMA_URL}. Is it installed and running?")
    except json.JSONDecodeError:
        raise LLMError("Ollama returned an unreadable response.")
    return data.get("message", {}).get("content", "").strip()


_BACKENDS = {"anthropic": _call_anthropic, "openai": _call_openai, "ollama": _call_ollama}


def generate(system, user):
    """Send the prompt to the chosen language model and return its text."""
    provider = pick_provider()
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    text = _BACKENDS[provider](model, system, user)
    if not text:
        raise LLMError("The language model returned an empty answer.")
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def answer_question(question, retriever, top_k=TOP_K, source=None):
    """
    Retrieve the best chunks for `question`, ask the language model, and return

        {"answer": str, "sources": [{"source", "page", "score", "text"}, ...]}

    `retriever` is a src.retriever.Retriever. Pass source="file.pdf" to search
    one PDF only. Raises LLMError if the language model cannot be reached.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("The question is empty.")

    results = [r for r in retriever.search(question, top_k=top_k, source=source) if r["score"] >= MIN_SCORE]
    if not results:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    answer = generate(SYSTEM_PROMPT, build_prompt(question, results))
    return {"answer": answer, "sources": results}


# ---------------------------------------------------------------------------
# Quick test from the command line (run from the project root):
#   python -m src.qa "your question"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(0)

    from src.retriever import Retriever

    try:
        output = answer_question(" ".join(sys.argv[1:]), Retriever())
    except LLMError as e:
        raise SystemExit(f"Error: {e}")

    print(output["answer"])
    if output["sources"]:
        print("\nSources:")
        for number, s in enumerate(output["sources"], start=1):
            print(f"  [{number}] {s['source']}, page {s['page']} (score {s['score']:.2f})")
