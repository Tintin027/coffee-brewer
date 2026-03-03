"""
Configuration for the Coffee RAG project.
Adjust these settings to match your setup.
"""

# --- Ollama Models ---
# Run `ollama pull <model>` before using
LLM_MODEL = "phi4-mini"                    # Generation model
EMBEDDING_MODEL = "nomic-embed-text"    # Embedding model
OLLAMA_BASE_URL = "http://localhost:11434"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME = "coffee_knowledge"

# --- Document Processing ---
DOCUMENTS_DIR = "data/documents"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between chunks

# --- Retrieval ---
TOP_K = 5                 # number of chunks to retrieve per query

# --- Generation ---
SYSTEM_PROMPT = """You are a knowledgeable coffee brewing assistant. 
Answer the user's question using ONLY the provided context. 
If the context doesn't contain enough information to answer, say so honestly.
Be specific with measurements, temperatures, and times when relevant.
Keep answers concise but helpful."""

PROMPT_TEMPLATE = """Context:
{context}

---

Question: {question}

Answer:"""
