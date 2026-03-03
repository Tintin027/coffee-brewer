"""
Gradio web interface for the Coffee RAG assistant.

Usage:
    python -m src.app
"""

import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query import CoffeeRAG

rag = CoffeeRAG()


def ask_coffee(question: str, history: list) -> str:
    """Handle a user question and return the RAG response."""
    if not question.strip():
        return "Please ask a coffee question!"

    result = rag.ask(question)
    answer = result["answer"]

    # append source info
    sources = set(c["source"] for c in result["sources"])
    if sources:
        answer += "\n\n---\n*Sources: " + ", ".join(sources) + "*"

    return answer


# ── Gradio Interface ─────────────────────────────────────────────────────────

with gr.Blocks(
    title="Coffee Brewing Assistant",
    theme=gr.themes.Soft(primary_hue="orange"),
) as app:
    gr.Markdown(
        """
        # ☕ Coffee Brewing Assistant
        Ask me anything about coffee — brewing methods, grind sizes, 
        troubleshooting, bean origins, and more. Powered by local AI.
        """
    )

    chatbot = gr.ChatInterface(
        fn=ask_coffee,
        examples=[
            "What's the ideal water temperature for pour over?",
            "My French press coffee tastes muddy, what should I do?",
            "What's the difference between Arabica and Robusta?",
            "Give me an AeroPress recipe for a single cup",
            "Why does my espresso have no crema?",
            "What grind size should I use for cold brew?",
        ],
        retry_btn=None,
        clear_btn="Clear",
    )

if __name__ == "__main__":
    app.launch()
