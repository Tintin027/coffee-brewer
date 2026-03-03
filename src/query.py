"""
Query engine: Retrieve relevant chunks → Build prompt → Generate answer.

Can be used as a library or run directly for CLI testing:
    python -m src.query "Why does my espresso taste sour?"
"""

import os
import sys

import chromadb
import ollama

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
)


class CoffeeRAG:
    """Lightweight RAG pipeline for coffee knowledge."""

    def __init__(self):
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[dict]:
        """Embed the question and retrieve the most relevant chunks."""
        response = ollama.embed(model=EMBEDDING_MODEL, input=question)
        query_embedding = response["embeddings"][0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i],
            })
        return chunks

    def generate(self, question: str, chunks: list[dict], stream: bool = False):
        """Build the augmented prompt and generate an answer."""
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in chunks
        )

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=stream,
        )

        if stream:
            return response  # returns a generator
        return response["message"]["content"]

    def ask(self, question: str, stream: bool = False):
        """Full RAG pipeline: retrieve → generate."""
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks, stream=stream)
        return {"answer": answer, "sources": chunks}


# ── CLI usage ────────────────────────────────────────────────────────────────

def main():
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    console = Console()
    rag = CoffeeRAG()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = console.input("[bold]Ask a coffee question:[/bold] ")

    console.print()

    with console.status("Thinking..."):
        result = rag.ask(question)

    # display answer
    console.print(Panel(Markdown(result["answer"]), title="☕ Answer", border_style="green"))

    # display sources
    console.print("\n[dim]Sources used:[/dim]")
    for chunk in result["sources"]:
        score = 1 - chunk["distance"]  # cosine similarity
        console.print(f"  [dim]• {chunk['source']} (similarity: {score:.2f})[/dim]")
    console.print()


if __name__ == "__main__":
    main()
