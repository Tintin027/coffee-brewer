"""
Ingestion pipeline: Load documents → Chunk → Embed → Store in ChromaDB.

Usage:
    python -m src.ingest                 # ingest all docs in data/documents/
    python -m src.ingest --reset         # wipe the DB and re-ingest
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import track

import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# allow running as `python -m src.ingest` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

console = Console()


# ── Document Loading ─────────────────────────────────────────────────────────

def load_documents(docs_dir: str) -> list[dict]:
    """Load all .md and .txt files from the documents directory."""
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        console.print(f"[red]Documents directory not found: {docs_dir}[/red]")
        return docs

    for filepath in sorted(docs_path.rglob("*")):
        if filepath.suffix in (".md", ".txt"):
            text = filepath.read_text(encoding="utf-8")
            docs.append({
                "text": text,
                "source": str(filepath.relative_to(docs_path)),
                "filename": filepath.name,
            })
            console.print(f"  Loaded [cyan]{filepath.name}[/cyan] ({len(text)} chars)")

    return docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "id": f"{doc['filename']}::chunk_{i}",
                "text": split,
                "metadata": {
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "chunk_index": i,
                },
            })

    return chunks


# ── Embedding + Storage ──────────────────────────────────────────────────────

def embed_and_store(chunks: list[dict]):
    """Embed chunks using Ollama and store in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # batch embed for efficiency
    batch_size = 50
    for start in track(range(0, len(chunks), batch_size), description="Embedding..."):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]

        # get embeddings from Ollama
        embeddings = []
        for text in texts:
            response = ollama.embed(model=EMBEDDING_MODEL, input=text)
            embeddings.append(response["embeddings"][0])

        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in batch],
        )

    return collection.count()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest coffee documents into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe existing DB before ingesting")
    args = parser.parse_args()

    console.print("\n[bold green]☕ Coffee RAG — Ingestion Pipeline[/bold green]\n")

    # optionally reset
    if args.reset and Path(CHROMA_PERSIST_DIR).exists():
        shutil.rmtree(CHROMA_PERSIST_DIR)
        console.print("[yellow]Cleared existing database.[/yellow]\n")

    # step 1: load
    console.print("[bold]1. Loading documents...[/bold]")
    docs = load_documents(DOCUMENTS_DIR)
    if not docs:
        console.print("[red]No documents found. Add .md or .txt files to data/documents/[/red]")
        return
    console.print(f"   Loaded {len(docs)} document(s)\n")

    # step 2: chunk
    console.print("[bold]2. Chunking documents...[/bold]")
    chunks = chunk_documents(docs)
    console.print(f"   Created {len(chunks)} chunks\n")

    # step 3: embed and store
    console.print("[bold]3. Embedding and storing...[/bold]")
    total = embed_and_store(chunks)
    console.print(f"\n   [green]✓ {total} chunks stored in ChromaDB[/green]\n")


if __name__ == "__main__":
    main()
