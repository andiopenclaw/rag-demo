"""
Step 2: Chunk articles, embed them, and store in ChromaDB.

This is the "indexing" phase of RAG — you do this once upfront.
The result is a persistent vector database that lives on disk.

Key concepts shown here:
  - Chunking: splitting long text into overlapping windows
  - Embeddings: turning text into semantic vectors (384 dimensions)
  - Vector DB: storing + indexing those vectors for fast retrieval
"""

import json
import os
import time
import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES_DIR = "articles"
DB_DIR = "chroma_db"
COLLECTION_NAME = "space_exploration"

# Chunking parameters — tunable!
CHUNK_SIZE = 400      # words per chunk
CHUNK_OVERLAP = 80    # words of overlap between chunks (preserves context at boundaries)


def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into overlapping word-window chunks.
    Overlap helps avoid cutting off a thought right at a boundary.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "id": f"{title.replace(' ', '_')}__chunk_{chunk_idx}",
            "text": chunk_text,
            "metadata": {
                "title": title,
                "chunk_index": chunk_idx,
                "word_start": start,
                "word_end": end,
                "word_count": len(chunk_words),
            }
        })

        chunk_idx += 1
        if end == len(words):
            break
        start += chunk_size - overlap  # slide forward with overlap

    return chunks


def build_index():
    # Load embedding model (runs locally, no API key needed)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    print("  → 384-dimensional sentence embeddings, ~80MB, very fast")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  ✓ Model loaded\n")

    # Set up ChromaDB (persisted to disk)
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Fresh start — delete collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
    )

    # Process each article
    article_files = [f for f in os.listdir(ARTICLES_DIR) if f.endswith(".json")]
    total_chunks = 0

    for filename in sorted(article_files):
        path = os.path.join(ARTICLES_DIR, filename)
        with open(path) as f:
            article = json.load(f)

        title = article["title"]
        text = article["text"]
        print(f"Processing: {title}")
        print(f"  Words: {len(text.split()):,}")

        # Chunk
        chunks = chunk_text(text, title)
        print(f"  Chunks: {len(chunks)} (size={CHUNK_SIZE}w, overlap={CHUNK_OVERLAP}w)")

        if not chunks:
            print("  ⚠️  Skipping (no content)\n")
            continue

        # Embed
        t0 = time.time()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        elapsed = time.time() - t0
        print(f"  Embedding time: {elapsed:.2f}s ({len(chunks)/elapsed:.1f} chunks/sec)")

        # Store in ChromaDB
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[c["metadata"] for c in chunks],
        )

        total_chunks += len(chunks)
        print()

    print(f"✅ Index built: {total_chunks} chunks from {len(article_files)} articles")
    print(f"   Stored in: ./{DB_DIR}/")
    return total_chunks


if __name__ == "__main__":
    build_index()
