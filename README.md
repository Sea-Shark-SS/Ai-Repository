# PDF Reader & Q&A with RAG

Ask questions about any PDF and get answers grounded in the document itself.

This project uses **Retrieval-Augmented Generation (RAG)**: instead of asking a language model to answer from memory, it first finds the most relevant passages in your PDF and gives only those to the model. The answers stay tied to the source and are less likely to be made up.

> 🚧 Replace every `TODO` in this file with the details of your own implementation.

---

## Features

- 📄 Upload and read PDF documents
- 🔎 Semantic search over the document content
- 💬 Natural-language question answering based on the PDF
- 📌 Answers backed by the retrieved passages (TODO: page numbers / source snippets)
- 🔁 Ask follow-up questions on the same document (TODO: remove if not supported)

## How it works

```
PDF ──► Extract text ──► Split into chunks ──► Create embeddings ──► Vector store
                                                                         │
Question ──► Embed question ──► Retrieve top-k similar chunks ◄──────────┘
                                        │
                                        ▼
                      Prompt = question + retrieved chunks
                                        │
                                        ▼
                           LLM generates the answer
```

1. **Ingestion:** text is extracted from the PDF and split into overlapping chunks.
2. **Indexing:** each chunk is converted into an embedding vector and stored in a vector database.
3. **Retrieval:** your question is embedded, and the most similar chunks are fetched.
4. **Generation:** the question and the retrieved chunks are sent to a language model, which writes the answer.

## Tech stack

| Part | Tool |
|---|---|
| Language | Python (TODO: version) |
| PDF parsing | TODO (e.g. PyPDF, PyMuPDF, pdfplumber) |
| Embeddings | TODO |
| Vector store | TODO (e.g. FAISS, Chroma) |
| LLM | TODO |
| Framework / UI | TODO (e.g. LangChain, LlamaIndex, Streamlit, Gradio) |

## Getting started

### Prerequisites

- Python 3.x (TODO: exact version)
- An API key for your LLM provider, if you use one (TODO)

### Installation

```bash
git clone https://github.com/Sea-Shark-SS/<your-repo-name>.git
cd <your-repo-name>

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root (TODO: adjust to your setup):

```env
API_KEY=your_key_here
```

> Never commit your `.env` file or API keys. Add `.env` to `.gitignore`.

### Run

```bash
python app.py                 # TODO: replace with your real command
# or: streamlit run app.py
```

## Usage

1. Start the app and upload a PDF.
2. Wait while the document is processed and indexed.
3. Type a question, for example:
   - *"What is the main conclusion of this paper?"*
   - *"Summarize section 3."*
   - *"What dataset did the authors use?"*
4. Read the answer and check the source passages shown with it.

## Project structure

```
.
├── app.py              # TODO: main entry point
├── requirements.txt
├── src/
│   ├── loader.py       # PDF loading and text extraction
│   ├── chunking.py     # text splitting
│   ├── embeddings.py   # embedding model
│   ├── retriever.py    # vector search
│   └── qa.py           # prompt building and answer generation
├── data/               # sample PDFs
└── README.md
```

> Adjust this tree to match your actual files.

## Example

| Question | Answer (abridged) |
|---|---|
| TODO: add a real question | TODO: add the answer your app gave |

You can also add a screenshot or GIF here:

```md
![Demo](assets/demo.png)
```

## Limitations

- Scanned PDFs (images of text) need OCR; plain text extraction will return little or nothing. (TODO: confirm for your version)
- Tables, equations and figures may not be extracted accurately.
- Answers are only as good as the retrieved passages. Very broad questions may miss relevant parts of a long document.
- Language models can still make mistakes, so verify important answers against the PDF.

## Future improvements

- [ ] Support multiple PDFs at once
- [ ] OCR for scanned documents
- [ ] Show page numbers with every answer
- [ ] Better retrieval (re-ranking, hybrid search)
- [ ] Evaluation of answer quality

## Author

**Choyon**, Computer Science and Engineering, East West University
GitHub: [@Sea-Shark-SS](https://github.com/Sea-Shark-SS)

## License

TODO: choose a license (for example MIT) and add a `LICENSE` file.
