# 📄 Doc Chatbot — a beginner-friendly AI project

Ask questions about any document and get answers, with the exact passages
the answer came from. Everything runs on your own computer — no API keys,
no accounts, no internet needed after the first run.

## How it works

This project is a tiny example of **RAG** (retrieval-augmented generation),
the same idea behind tools that let chatbots answer from your own files:

1. **Chunk** — the document is split into small overlapping pieces
2. **Embed** — each piece is converted into a list of numbers (a *vector*)
   that captures its meaning, using a small local model
3. **Retrieve** — your question is converted the same way, and the pieces
   with the closest vectors are picked out
4. **Answer** — a question-answering model reads those pieces and extracts
   the answer

All of this lives in `rag.py` (about 100 lines, well commented).
`app.py` is the Gradio web interface on top of it.

## Setup

You need Python 3.10 or newer.

```bash
cd ai-doc-chatbot
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# PyTorch (CPU version — smaller download, fine for this project)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run it

Web app:

```bash
python app.py
```

Then open the URL it prints (usually http://127.0.0.1:7860) in your browser.
A sample document about dragon care is pre-loaded, so try asking
*"What do dragons eat?"* — then upload your own `.txt`, `.md`, or `.pdf`.

Command-line version:

```bash
python rag.py
```

## Ideas to extend it

- **Chat history** — keep a list of past Q&A pairs and show them in the UI
- **Better answers** — swap the QA model for a generative one (e.g. connect
  the OpenAI API) so it can summarize, not just extract quotes
- **Bigger documents** — replace the in-memory list with a real vector
  database like FAISS or Chroma
- **More file types** — add `.docx` support with `python-docx`

## Troubleshooting

- **First run is slow** — the models (~400 MB) download from Hugging Face
  once, then are cached. Be patient, it only happens once.
- **`torch` install issues** — if the CPU index doesn't work for your
  platform, just run `pip install torch` instead (it's a bigger download).
