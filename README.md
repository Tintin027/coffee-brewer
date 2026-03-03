# ☕ Coffee RAG — Local Brewing Assistant

A lightweight Retrieval-Augmented Generation (RAG) project that answers coffee brewing questions using local models via Ollama. No API keys needed — everything runs on your machine.

## Architecture

```
Question → Embed (nomic-embed-text) → Retrieve from ChromaDB → Augment prompt → Generate (Llama 3)
```

```
coffee-rag/
├── data/
│   ├── documents/          # Your coffee knowledge base (.md, .txt files)
│   │   ├── 01_brewing_methods.md
│   │   ├── 02_troubleshooting.md
│   │   ├── 03_beans_and_origins.md
│   │   └── 04_grind_guide.md
│   └── chroma_db/          # Vector database (auto-generated)
├── src/
│   ├── config.py           # All settings in one place
│   ├── ingest.py           # Document loading → chunking → embedding → storage
│   ├── query.py            # Retrieval → prompt building → generation
│   └── app.py              # Gradio web interface
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# or download from https://ollama.com
```

### 2. Pull the models

```bash
ollama pull phi4-mini
ollama pull nomic-embed-text
```

### 3. Install Python dependencies

```bash
cd coffee-rag
pip install -r requirements.txt
```

### 4. Ingest the documents

```bash
python -m src.ingest
```

This reads all `.md` and `.txt` files from `data/documents/`, chunks them, generates embeddings, and stores everything in ChromaDB.

To re-ingest from scratch:
```bash
python -m src.ingest --reset
```

### 5. Start asking questions

**CLI mode:**
```bash
python -m src.query "Why does my espresso taste sour?"
```

**Web UI:**
```bash
python -m src.app
```
Then open http://localhost:7860 in your browser.

## Adding Your Own Knowledge

Drop any `.md` or `.txt` files into `data/documents/` and re-run ingestion:

```bash
python -m src.ingest --reset
```

Ideas for content to add:
- Your personal brew recipes and notes
- Specific equipment manuals or guides
- Articles from coffee blogs (saved as text)
- Tasting notes from beans you've tried
- Roasting profiles and notes

## Configuration

All settings live in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama3` | Ollama model for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `CHUNK_SIZE` | `500` | Characters per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |

## Next Steps / Ideas

- [ ] Implement re-ranking (e.g., with a cross-encoder) for better retrieval
- [ ] Add chat history / multi-turn conversation support
- [ ] Experiment with different chunk sizes and overlap
- [ ] Try different models (Mistral, Phi-3, Gemma) and compare quality
- [ ] Add metadata filtering (e.g., "only search brewing method docs")
- [ ] Build a recipe database with structured data alongside the RAG
