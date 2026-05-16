"""
Step 3: Query the RAG system.

This is the "retrieval + generation" phase:
  1. Embed the user's question
  2. Find the most semantically similar chunks (retrieval)
  3. Build a prompt: question + retrieved context
  4. Send to an LLM (OpenAI) or display retrieved context alone

Set OPENAI_API_KEY in your environment to enable generation.
Without it, the script shows retrieval results only — still very useful
for demonstrating the concept.
"""

import os
import sys
import time
import json
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION_NAME = "space_exploration"
TOP_K = 3  # how many chunks to retrieve


def retrieve(query: str, collection, model, top_k: int = TOP_K):
    """Embed the query and find the closest chunks."""
    t0 = time.time()
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    elapsed = time.time() - t0

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "chunk_index": meta["chunk_index"],
            "similarity": 1 - dist,  # cosine distance → similarity
        })

    return chunks, elapsed


def generate_with_openai(query: str, context_chunks: list) -> str:
    """Use OpenAI to generate a grounded answer from retrieved context."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    context = "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['text']}" for c in context_chunks
    )

    system_prompt = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say so honestly.
Be concise and specific. Cite sources by article title."""

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    elapsed = time.time() - t0

    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    return answer, elapsed, tokens_used


def pretty_print_result(query: str, chunks: list, retrieval_time: float, answer=None, gen_time=None, tokens=None):
    width = 72
    print("\n" + "=" * width)
    print(f"  QUERY: {query}")
    print("=" * width)

    print(f"\n🔍 RETRIEVAL  ({retrieval_time*1000:.0f}ms | top {len(chunks)} chunks)\n")
    for i, chunk in enumerate(chunks, 1):
        sim_bar = "█" * int(chunk["similarity"] * 20)
        print(f"  [{i}] {chunk['title']}  (similarity: {chunk['similarity']:.3f}  {sim_bar})")
        # Show first 200 chars of chunk
        preview = chunk["text"][:220].replace("\n", " ")
        if len(chunk["text"]) > 220:
            preview += "…"
        print(f"      {preview}")
        print()

    if answer:
        print(f"🤖 GENERATED ANSWER  ({gen_time*1000:.0f}ms | {tokens} tokens)\n")
        for line in answer.strip().split("\n"):
            print(f"  {line}")
        print()
    else:
        print("  ℹ️  (Set OPENAI_API_KEY to enable answer generation)")
        print()

    print("=" * width + "\n")


def main():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    mode = "Retrieval + Generation (OpenAI)" if has_openai else "Retrieval only (no OpenAI key)"
    print(f"Mode: {mode}")
    print(f"Loaded collection: {COLLECTION_NAME} ({collection.count()} chunks)\n")

    # Demo queries — great for showing colleagues
    demo_queries = [
        "How many astronauts walked on the Moon during the Apollo program?",
        "What are the primary scientific goals of the James Webb Space Telescope?",
        "When did SpaceX first successfully land a Falcon 9 booster?",
        "How does the International Space Station get its power?",
        "What rovers have successfully operated on Mars?",
    ]

    # If called with an argument, use that as the query
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = demo_queries

    for query in queries:
        chunks, retrieval_time = retrieve(query, collection, model)

        answer_data = None
        if has_openai:
            result = generate_with_openai(query, chunks)
            if result:
                answer, gen_time, tokens = result
                answer_data = (answer, gen_time, tokens)

        if answer_data:
            pretty_print_result(query, chunks, retrieval_time, *answer_data)
        else:
            pretty_print_result(query, chunks, retrieval_time)

        if len(queries) > 1:
            time.sleep(0.3)


if __name__ == "__main__":
    main()
