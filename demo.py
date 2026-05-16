"""
🌱 RAG Demo — All-in-one runner for colleague presentations.

Run this to walk through the entire pipeline step by step with
commentary, timing, and educational output.

Usage:
    python demo.py              # full demo with built-in questions
    python demo.py --rebuild    # re-fetch articles and rebuild index
    python demo.py "your question here"
"""

import os
import sys
import time


def banner(title: str, emoji: str = ""):
    width = 72
    line = "─" * width
    print(f"\n{line}")
    print(f"  {emoji}  {title}")
    print(f"{line}\n")


def explain(text: str):
    """Print an educational explanation block."""
    for line in text.strip().split("\n"):
        print(f"  💬 {line.strip()}")
    print()


def main():
    rebuild = "--rebuild" in sys.argv
    custom_query = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            custom_query = arg

    # ─── INTRO ──────────────────────────────────────────────────────────────
    banner("RAG DEMO — Retrieval-Augmented Generation", "🚀")
    explain("""
        RAG is a technique that grounds an LLM's answers in real documents.
        Instead of relying on what the model "remembers" from training,
        we:
          1. Build a searchable vector index from your documents
          2. At query time, retrieve the most relevant chunks
          3. Feed those chunks as context to the LLM
          4. Get an answer grounded in YOUR data — not hallucinations
        
        Today's demo: 5 Wikipedia articles about Space Exploration.
    """)

    # ─── STEP 1: FETCH ──────────────────────────────────────────────────────
    articles_exist = os.path.exists("articles") and len(
        [f for f in os.listdir("articles") if f.endswith(".json")]
    ) > 0
    db_exists = os.path.exists("chroma_db")

    if rebuild or not articles_exist:
        banner("STEP 1 — Fetching Wikipedia Articles", "📖")
        explain("""
            We're pulling 5 articles from Wikipedia using their public API.
            In production, this would be your own documents, PDFs, databases, etc.
        """)
        from fetch_articles import main as fetch
        fetch()
    else:
        banner("STEP 1 — Articles Already Fetched", "📖")
        files = [f for f in os.listdir("articles") if f.endswith(".json")]
        print(f"  Using {len(files)} cached articles (run with --rebuild to refresh)\n")
        import json
        for f in sorted(files):
            with open(f"articles/{f}") as fh:
                a = json.load(fh)
            print(f"  • {a['title']}  ({a['word_count']:,} words)")
        print()

    # ─── STEP 2: INDEX ──────────────────────────────────────────────────────
    if rebuild or not db_exists:
        banner("STEP 2 — Building the Vector Index", "🗄️")
        explain("""
            This is where the magic starts. We:
              • Split each article into overlapping chunks (~400 words each)
              • Run each chunk through a sentence embedding model
                (all-MiniLM-L6-v2: 384 dimensions, runs locally, free)
              • Store the vectors in ChromaDB (a local vector database)
            
            Chunking with overlap prevents losing context at boundaries.
            This whole step runs ONCE — then queries are fast.
        """)
        t0 = time.time()
        from build_index import build_index
        total_chunks = build_index()
        elapsed = time.time() - t0
        print(f"\n  ⏱  Total indexing time: {elapsed:.1f}s for {total_chunks} chunks\n")
    else:
        banner("STEP 2 — Index Already Built", "🗄️")
        import chromadb as _chromadb
        _client = _chromadb.PersistentClient(path="chroma_db")
        _col = _client.get_collection("space_exploration")
        print(f"  Using existing index: {_col.count()} chunks in ChromaDB")
        print(f"  (run with --rebuild to re-index)\n")

    # ─── STEP 3: QUERY ──────────────────────────────────────────────────────
    banner("STEP 3 — Querying the RAG System", "🔍")
    explain("""
        Now the fun part! For each question:
          1. We embed the question into the same vector space
          2. ChromaDB finds the nearest chunks (cosine similarity)
          3. We show you exactly WHICH chunks were retrieved and HOW similar
          4. If you have an OpenAI API key, we generate a grounded answer
        
        Notice the similarity scores — they show the model's confidence
        in relevance. Anything above 0.5 is usually solid.
    """)

    from query import retrieve, generate_with_openai, pretty_print_result
    from sentence_transformers import SentenceTransformer
    import chromadb

    print("  Loading embedding model...", end=" ", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓\n")

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("space_exploration")

    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    if has_openai:
        print("  🔑 OpenAI key detected — will generate answers via gpt-4o-mini\n")
    else:
        print("  ℹ️  No OPENAI_API_KEY set — showing retrieval results only")
        print("     Set it to see full answer generation:\n")
        print("     export OPENAI_API_KEY=sk-...\n")

    if custom_query:
        queries = [custom_query]
    else:
        queries = [
            "How many astronauts walked on the Moon during the Apollo program?",
            "What are the primary scientific goals of the James Webb Space Telescope?",
            "When did SpaceX first successfully land a Falcon 9 booster?",
        ]

    for query in queries:
        input(f"  ▶  Press Enter to run: \"{query}\"\n     ")
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

        time.sleep(0.2)

    # ─── WRAP UP ────────────────────────────────────────────────────────────
    banner("WHAT YOU JUST SAW", "🎓")
    explain("""
        The full RAG pipeline, end to end:
        
          📖 Ingest    Wikipedia articles → text
          ✂️  Chunk     Split into ~400-word windows with 80-word overlap
          🔢 Embed     SentenceTransformer → 384-dim vectors (local, free)
          🗄️  Store     ChromaDB persistent vector database
          🔍 Retrieve  Cosine similarity search in <50ms
          🤖 Generate  LLM answers grounded in retrieved context (optional)
        
        Where to go from here:
          • Swap in your own documents (PDFs, Notion, Confluence, etc.)
          • Try different chunk sizes — smaller = more precise, larger = more context
          • Add metadata filters (e.g., only search articles from Q1)
          • Evaluate with RAGAS or LangSmith for production use
          • Scale up with Pinecone, Weaviate, or pgvector for larger corpora
    """)


if __name__ == "__main__":
    main()
