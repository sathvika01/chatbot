"""
The brain of the Doc Chatbot: a tiny retrieval-augmented generation (RAG) pipeline.

How it works, in 4 beginner-friendly steps:
  1. CHUNK    - the document is split into small overlapping pieces of text
  2. EMBED    - each chunk is turned into a list of numbers (a "vector") that
                captures its meaning, using a small local model
  3. RETRIEVE - your question is turned into a vector too, and we find the
                chunks whose vectors are closest (cosine similarity)
  4. ANSWER   - the best chunks are handed to a question-answering model,
                which reads them and pulls out the answer

Everything runs locally on your machine. No API keys, no accounts, no internet
needed after the models download once.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@dataclass
class Passage:
    """One retrieved chunk of the document, with its relevance score."""
    text: str
    score: float


class DocChatbot:
    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qa_model: str = "distilbert-base-cased-distilled-squad",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        print(f"Loading embedding model: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)

        print(f"Loading question-answering model: {qa_model}")
        self.qa = pipeline("question-answering", model=qa_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    # ---------- loading a document ----------

    def load_text(self, text: str) -> None:
        """Split `text` into chunks and pre-compute their embeddings."""
        self.chunks = self._chunk(text)
        raw = self.embedder.encode(self.chunks, convert_to_numpy=True, show_progress_bar=False)
        # Normalize so that a dot product == cosine similarity
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-9
        self.embeddings = raw / norms
        print(f"Indexed {len(self.chunks)} chunks.")

    def _chunk(self, text: str) -> list[str]:
        """Cut text into overlapping windows of characters."""
        step = self.chunk_size - self.chunk_overlap
        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
        return [c for c in chunks if c.strip()]

    # ---------- answering a question ----------

    def answer(self, question: str, top_k: int = 3) -> tuple[str, float, list[Passage]]:
        """Find the most relevant chunks and answer from them."""
        if not self.chunks:
            raise ValueError("No document loaded yet. Call load_text() first.")

        # Embed the question the same way the chunks were embedded
        q = self.embedder.encode([question], convert_to_numpy=True, show_progress_bar=False)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

        # Cosine similarity between the question and every chunk
        scores = (self.embeddings @ q.T).ravel()
        best = np.argsort(scores)[::-1][:top_k]
        passages = [Passage(self.chunks[i], float(scores[i])) for i in best]

        # Let the QA model read the retrieved passages and answer
        context = "\n\n".join(p.text for p in passages)
        result = self.qa(question=question, context=context)
        return result["answer"], float(result.get("score", 0.0)), passages


if __name__ == "__main__":
    # Quick command-line demo: python rag.py
    bot = DocChatbot()
    with open("sample_doc.txt", encoding="utf-8") as f:
        bot.load_text(f.read())

    print("\nAsk questions about the sample document (empty line to quit).")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            break
        answer, confidence, _ = bot.answer(question)
        print(f"Bot ({confidence:.0%} confident): {answer}")
