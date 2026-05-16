# 🚀 RAG Demo — Retrieval-Augmented Generation

A hands-on demo of RAG built for a 20-minute colleague presentation.
Uses Wikipedia articles about Space Exploration as the knowledge base.

---

## What is RAG?

**Retrieval-Augmented Generation** is a technique that makes LLMs smarter about *your* data.

The problem with plain LLMs:
- They only know what was in their training data (cutoff date)
- They hallucinate when asked specific facts they're uncertain about
- They can't access your internal documents

RAG solves this by:
1. **Indexing** your documents into a vector database
2. **Retrieving** the most relevant passages at query time
3. **Grounding** the LLM's answer in those passages

```
YOUR DOCS ──► CHUNKS ──► EMBEDDINGS ──► VECTOR DB
                                            │
USER QUERY ──► EMBED ──► SIMILARITY SEARCH ─┘
                                            │
                                     TOP-K CHUNKS
                                            │
                              LLM(query + context) ──► ANSWER
```

---

## Quick Start

```bash
# 1. Clone / enter the project
cd rag-demo

# 2. Activate the virtual environment
source venv/bin/activate

# 3. Run the full interactive demo
python demo.py

# 4. (Optional) Enable answer generation
export OPENAI_API_KEY=sk-...
python demo.py

# 5. Ask your own question
python demo.py "What fuel does the Falcon 9 use?"
```

---

## How it Works — Step by Step

### Step 1: Fetch Articles (`fetch_articles.py`)
Pulls 5 Wikipedia articles via the public API. In production, this is replaced with your document ingestion pipeline (PDFs, databases, web crawlers, etc.).

**Articles used:**
- Apollo Program
- James Webb Space Telescope
- SpaceX
- International Space Station
- Mars Exploration

### Step 2: Build the Index (`build_index.py`)

**Chunking**
Each article is split into overlapping windows of ~400 words:
```
[───── chunk 1 ─────]
              [───── chunk 2 ─────]
                            [───── chunk 3 ─────]
```
Overlap (80 words) prevents losing context at boundaries.

**Embedding**
Each chunk is converted to a 384-dimensional vector using
`sentence-transformers/all-MiniLM-L6-v2` — a fast, free, local model.

Words with similar meaning end up close together in this vector space:
- "Moon landing" ↔ "lunar surface" ↔ "Apollo 11" → all nearby
- "rocket fuel" ↔ "propellant" → nearby
- "telescope mirror" ↔ "orbital mechanics" → far apart

**Storage**
Vectors + original text stored in **ChromaDB** — a lightweight
persistent vector database that lives on disk.

### Step 3: Query (`query.py`)

At query time:
1. Embed the user's question (same model, <10ms)
2. Find top-K most similar chunks by cosine similarity (<50ms)
3. Build a prompt: `system + context chunks + question`
4. LLM generates a grounded answer

---

## Architecture Choices (and alternatives)

| Component | This Demo | Production Alternatives |
|-----------|-----------|------------------------|
| Embeddings | `all-MiniLM-L6-v2` (local, free) | OpenAI `text-embedding-3-small`, Cohere |
| Vector DB | ChromaDB (local) | Pinecone, Weaviate, pgvector, Qdrant |
| LLM | GPT-4o-mini | GPT-4o, Claude, Gemini, local Llama |
| Chunking | Fixed-size + overlap | Semantic chunking, document-structure aware |
| Retrieval | Top-K cosine similarity | MMR, hybrid (BM25 + vector), re-ranking |

---

## Performance Profile

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Indexing 5 articles | ~15–30s | One-time cost |
| Query embedding | <10ms | |
| Vector search (ChromaDB) | <50ms | Scales well to ~1M chunks |
| LLM generation | 1–5s | Network dependent |

---

## Why RAG beats fine-tuning for most use cases

| | Fine-tuning | RAG |
|---|---|---|
| Update knowledge | Retrain ($$$) | Re-index (fast, cheap) |
| Cite sources | No | Yes |
| Control | Limited | Full |
| Setup cost | High | Low |
| Hallucinations | Still happens | Greatly reduced |

---

## Going Further

- **Evaluation**: Use [RAGAS](https://github.com/explodinggradients/ragas) to measure faithfulness, relevancy, context recall
- **Hybrid search**: Combine BM25 (keyword) + vector (semantic) for better recall
- **Re-ranking**: Use a cross-encoder to re-score retrieved chunks
- **Metadata filtering**: Filter by date, author, department before similarity search
- **Streaming**: Stream LLM output for snappier UX
- **Conversation**: Add chat history for multi-turn Q&A

---

*Built with: Python · sentence-transformers · ChromaDB · OpenAI (optional)*
