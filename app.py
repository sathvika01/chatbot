"""Web UI for the Doc Chatbot, built with Gradio. Run with: python app.py"""

import gradio as gr
from pypdf import PdfReader

from rag import DocChatbot

print("Loading AI models (first run downloads them, this can take a few minutes)...")
bot = DocChatbot()
print("Models ready.")

# Pre-load the sample document so the app works immediately.
with open("sample_doc.txt", encoding="utf-8") as f:
    bot.load_text(f.read())


def read_file(file) -> str:
    """Read text out of an uploaded .txt, .md, or .pdf file."""
    path = file if isinstance(file, str) else file.name
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def load_document(file):
    if file is None:
        return "Please choose a file first."
    text = read_file(file).strip()
    if not text:
        return "Couldn't read any text from that file."
    bot.load_text(text)
    return f"Loaded! The document was split into {len(bot.chunks)} chunks."


def ask(question):
    if not question or not question.strip():
        return "Type a question first!", ""
    if not bot.chunks:
        return "Load a document first.", ""
    answer, confidence, passages = bot.answer(question.strip())
    sources = "\n\n---\n\n".join(
        f"**Passage {i + 1}** (match score: {p.score:.2f})\n\n{p.text.strip()[:600]}"
        for i, p in enumerate(passages)
    )
    return f"{answer}\n\n_(confidence: {confidence:.0%})_", sources


with gr.Blocks(title="Doc Chatbot") as demo:
    gr.Markdown(
        "# 📄 Doc Chatbot\n"
        "Upload a document, then ask questions about it. "
        "A sample document is already loaded, so you can try it right away."
    )
    with gr.Row():
        file_input = gr.File(
            label="Document (.txt, .md, .pdf)",
            file_types=[".txt", ".md", ".pdf"],
        )
        load_btn = gr.Button("Load document", variant="primary")
    status = gr.Textbox(
        label="Status", value="Sample document loaded.", interactive=False
    )
    question = gr.Textbox(
        label="Your question", placeholder="e.g. What do dragons eat?"
    )
    ask_btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer")
    sources_md = gr.Markdown(label="Retrieved passages")

    load_btn.click(load_document, inputs=file_input, outputs=status)
    ask_btn.click(ask, inputs=question, outputs=[answer, sources_md])
    question.submit(ask, inputs=question, outputs=[answer, sources_md])

demo.launch()
