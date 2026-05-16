"""
RAG Demo — Chainlit App
Chat interface for querying the knowledge base.

Run with:
    chainlit run app.py

With OpenAI (for generated answers):
    export OPENAI_API_KEY=sk-...
    chainlit run app.py
"""

import os
import time
import chainlit as cl
from sentence_transformers import SentenceTransformer
import chromadb

try:
    from openai import AsyncOpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

DB_DIR = "chroma_db"
COLLECTION_NAME = "space_exploration"
TOP_K = 3

# ── Load once at startup ──────────────────────────────────────────────────────

print("Loading embedding model...")
_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
try:
    _collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION_NAME)
    print(f"✓ Knowledge base ready: {_collection.count()} chunks")
except Exception as e:
    _collection = None
    print(f"✗ Could not load knowledge base: {e}")
    print("  Run: python build_index.py first")


# ── Helpers ───────────────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 1000  # characters — prevents prompt stuffing


def has_openai() -> bool:
    """Check at call time so env changes after startup are picked up."""
    return _openai_available and bool(os.environ.get("OPENAI_API_KEY"))


def get_openai_key() -> str:
    """Safely retrieve the OpenAI key — raises clearly if missing."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY disappeared from environment")
    return key


# ── Chainlit handlers ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    if _collection is None:
        await cl.Message(
            content="⚠️ Knowledge base not loaded. Run `python build_index.py` first."
        ).send()
        return

    mode = "🤖 Retrieval + Generation (OpenAI)" if has_openai() else "🔍 Retrieval only"
    await cl.Message(
        content=(
            f"## 🚀 RAG Demo — Space Exploration\n\n"
            f"**{_collection.count()} chunks** indexed · **Mode:** {mode}\n\n"
            f"Ask anything about the **Apollo program**, **James Webb Space Telescope**, "
            f"**SpaceX**, or the **ISS**!\n\n"
            + ("" if has_openai() else
               "_Set `OPENAI_API_KEY` to enable generated answers. "
               "Right now I'll show retrieved sources._")
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    if _collection is None:
        await cl.Message(content="⚠️ Knowledge base not loaded.").send()
        return

    query = message.content.strip()

    # Basic input guard — prevent prompt stuffing / very long inputs
    if len(query) > MAX_QUERY_LENGTH:
        await cl.Message(
            content=f"⚠️ Query too long ({len(query)} chars). Please keep it under {MAX_QUERY_LENGTH} characters."
        ).send()
        return
    if not query:
        return

    # ── Step 1: Embed the query ───────────────────────────────────────────────
    async with cl.Step(name="1️⃣  Embedding query", type="tool") as step:
        t0 = time.time()
        embedding = _model.encode([query])[0]
        embed_time = time.time() - t0
        step.output = (
            f"Converted query into a **{len(embedding)}-dimensional vector** "
            f"in `{embed_time * 1000:.0f}ms` using `all-MiniLM-L6-v2`.\n\n"
            f"This vector represents the *meaning* of your question in numeric form."
        )

    # ── Step 2: Retrieve from ChromaDB ────────────────────────────────────────
    async with cl.Step(name="2️⃣  Searching knowledge base", type="retrieval") as step:
        t0 = time.time()
        results = _collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        retrieval_time = time.time() - t0

        chunks = [
            {
                "text": doc,
                "title": meta["title"],
                "similarity": round(1 - dist, 3),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

        step.output = (
            f"Searched **{_collection.count()} chunks** in ChromaDB using cosine similarity.\n"
            f"Retrieved top {TOP_K} matches in `{retrieval_time * 1000:.0f}ms`:\n\n"
            + "\n".join(
                f"- **[{i}]** `{c['similarity']:.3f}` {'█' * int(c['similarity'] * 15)}  {c['title']}"
                for i, c in enumerate(chunks, 1)
            )
        )

    source_elements = [
        cl.Text(
            name=f"[{i}] {c['title']}  ·  {c['similarity']:.3f}",
            content=c["text"],
            display="inline",
        )
        for i, c in enumerate(chunks, 1)
    ]

    # ── Step 3: Generate answer ───────────────────────────────────────────────
    if has_openai():
        oai = AsyncOpenAI(api_key=get_openai_key())
        context = "\n\n---\n\n".join(
            f"[Source: {c['title']}]\n{c['text']}" for c in chunks
        )

        msg = cl.Message(content="", elements=source_elements)
        await msg.send()

        async with cl.Step(name="3️⃣  Generating answer", type="llm") as step:
            step.input = (
                f"Sending **{len(context.split())} words** of context to `gpt-4o-mini`.\n\n"
                f"Grounded on {len(chunks)} source chunks."
            )
            stream = await oai.chat.completions.create(  # noqa: S106
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer using ONLY the provided context. "
                            "If the context lacks enough info, say so. "
                            "Be concise and cite source titles."
                        ),
                    },
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                ],
                temperature=0.2,
                max_tokens=500,
                stream=True,
            )
            full_response = ""
            async for part in stream:
                token = part.choices[0].delta.content or ""
                full_response += token
                await msg.stream_token(token)
            step.output = full_response

        await msg.stream_token(
            f"\n\n---\n_⚡ {embed_time*1000:.0f}ms embed · "
            f"{retrieval_time*1000:.0f}ms retrieve · gpt-4o-mini_"
        )
        await msg.update()

    else:
        await cl.Message(
            content=(
                f"_No OpenAI key — showing retrieved sources:_\n\n"
                + "\n".join(
                    f"**[{i}]** `{c['similarity']:.3f}` {c['title']}"
                    for i, c in enumerate(chunks, 1)
                )
                + f"\n\n_⚡ {embed_time*1000:.0f}ms embed · {retrieval_time*1000:.0f}ms retrieve_"
            ),
            elements=source_elements,
        ).send()
