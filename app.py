"""
RAG Demo — Chainlit App
Chat interface for querying the knowledge base.

Supports three LLM providers, selectable per-session via the UI (⚙️):
  • Claude     — set ANTHROPIC_API_KEY
  • OpenAI     — set OPENAI_API_KEY
  • Azure AI Foundry (Codex) — set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT

Run with:
    chainlit run app.py --host 0.0.0.0 --port 8000
"""

import os
import time
import chainlit as cl
from chainlit.input_widget import Select
from sentence_transformers import SentenceTransformer
import chromadb

# ── Optional provider SDKs ────────────────────────────────────────────────────
try:
    import anthropic
    _anthropic_ok = True
except ImportError:
    _anthropic_ok = False

try:
    from openai import AsyncOpenAI
    _openai_ok = True
except ImportError:
    _openai_ok = False

# ── Constants ─────────────────────────────────────────────────────────────────
DB_DIR = "chroma_db"
COLLECTION_NAME = "space_exploration"
TOP_K = 3
MAX_QUERY_LENGTH = 1000  # chars — prevents prompt stuffing

SYSTEM_PROMPT = (
    "Answer using ONLY the provided context. "
    "If the context lacks enough information, say so honestly. "
    "Be concise and cite source titles."
)

# ── Provider catalogue ────────────────────────────────────────────────────────
# Each entry: display label → (provider_key, model_id)
# Only entries whose provider is available (key set) are shown in the UI.

MODELS = {
    # Claude
    "Claude — Haiku (fast)":       ("claude",  "claude-haiku-4-5"),
    "Claude — Sonnet (balanced)":  ("claude",  "claude-sonnet-4-5"),
    "Claude — Opus (best)":        ("claude",  "claude-opus-4-5"),
    # OpenAI
    "OpenAI — GPT-4o mini (fast)": ("openai",  "gpt-4o-mini"),
    "OpenAI — GPT-4o":             ("openai",  "gpt-4o"),
    # Azure AI Foundry
    "Foundry — o3-mini (Codex)":   ("azure",   "o3-mini"),
    "Foundry — GPT-4o":            ("azure",   "gpt-4o"),
}

DEFAULT_PROVIDER_ORDER = ["claude", "openai", "azure"]
DEFAULT_MODEL_LABEL = {
    "claude": "Claude — Sonnet (balanced)",
    "openai": "OpenAI — GPT-4o mini (fast)",
    "azure":  "Foundry — o3-mini (Codex)",
}


def provider_available(provider: str) -> bool:
    if provider == "claude":
        return _anthropic_ok and bool(os.environ.get("ANTHROPIC_API_KEY"))
    if provider == "openai":
        return _openai_ok and bool(os.environ.get("OPENAI_API_KEY"))
    if provider == "azure":
        return _openai_ok and bool(os.environ.get("AZURE_OPENAI_API_KEY")) \
               and bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))
    return False


def available_model_labels() -> list[str]:
    """Return only labels for providers that have keys configured."""
    return [
        label for label, (provider, _) in MODELS.items()
        if provider_available(provider)
    ]


def default_model_label() -> str | None:
    """Pick the first available model from the preferred provider order."""
    for provider in DEFAULT_PROVIDER_ORDER:
        if provider_available(provider):
            return DEFAULT_MODEL_LABEL[provider]
    return None


def get_key(provider: str) -> str:
    """Safely retrieve the API key for a provider."""
    env = {
        "claude": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure":  "AZURE_OPENAI_API_KEY",
    }[provider]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"{env} is not set")
    return key


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


# ── Generation helpers ────────────────────────────────────────────────────────

async def stream_claude(model: str, context: str, query: str, msg: cl.Message):
    client = anthropic.AsyncAnthropic(api_key=get_key("claude"))
    async with client.messages.stream(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    ) as stream:
        async for token in stream.text_stream:
            await msg.stream_token(token)


async def stream_openai(model: str, context: str, query: str, msg: cl.Message, azure: bool = False):
    if azure:
        client = AsyncOpenAI(
            api_key=get_key("azure"),
            base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai",
            default_query={"api-version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")},
        )
    else:
        client = AsyncOpenAI(api_key=get_key("openai"))

    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=500,
        stream=True,
    )
    async for part in stream:
        token = (part.choices[0].delta.content or "") if part.choices else ""
        if token:
            await msg.stream_token(token)


# ── Chainlit handlers ─────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    if _collection is None:
        await cl.Message(
            content="⚠️ Knowledge base not loaded. Run `python build_index.py` first."
        ).send()
        return

    labels = available_model_labels()
    default = default_model_label()

    if not labels:
        # No provider keys configured — retrieval-only mode
        cl.user_session.set("model_label", None)
        await cl.Message(
            content=(
                f"## 🚀 RAG Demo — Space Exploration\n\n"
                f"**{_collection.count()} chunks** indexed · **Mode:** 🔍 Retrieval only\n\n"
                f"Ask anything about the **Apollo program**, **James Webb Space Telescope**, "
                f"**SpaceX**, or the **ISS**!\n\n"
                "_No LLM keys found — set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or "
                "`AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` to enable generated answers._"
            )
        ).send()
        return

    # Build and send settings panel
    settings = await cl.ChatSettings([
        Select(
            id="model_label",
            label="🤖 LLM Model",
            values=labels,
            initial_value=default,
        )
    ]).send()

    cl.user_session.set("model_label", settings["model_label"])

    provider, model_id = MODELS[settings["model_label"]]
    await cl.Message(
        content=(
            f"## 🚀 RAG Demo — Space Exploration\n\n"
            f"**{_collection.count()} chunks** indexed · "
            f"**Model:** `{model_id}`\n\n"
            f"Ask anything about the **Apollo program**, **James Webb Space Telescope**, "
            f"**SpaceX**, or the **ISS**!\n\n"
            f"_Change model anytime via the ⚙️ settings icon._"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    label = settings.get("model_label")
    cl.user_session.set("model_label", label)
    if label:
        _, model_id = MODELS[label]
        await cl.Message(content=f"✅ Switched to `{model_id}`").send()


@cl.on_message
async def on_message(message: cl.Message):
    if _collection is None:
        await cl.Message(content="⚠️ Knowledge base not loaded.").send()
        return

    query = message.content.strip()

    if not query:
        return
    if len(query) > MAX_QUERY_LENGTH:
        await cl.Message(
            content=f"⚠️ Query too long ({len(query)} chars). Please keep it under {MAX_QUERY_LENGTH} characters."
        ).send()
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
    model_label = cl.user_session.get("model_label")

    if not model_label:
        # Retrieval-only fallback
        await cl.Message(
            content=(
                "_No LLM configured — showing retrieved sources:_\n\n"
                + "\n".join(
                    f"**[{i}]** `{c['similarity']:.3f}` {c['title']}"
                    for i, c in enumerate(chunks, 1)
                )
                + f"\n\n_⚡ {embed_time*1000:.0f}ms embed · {retrieval_time*1000:.0f}ms retrieve_"
            ),
            elements=source_elements,
        ).send()
        return

    provider, model_id = MODELS[model_label]
    context = "\n\n---\n\n".join(
        f"[Source: {c['title']}]\n{c['text']}" for c in chunks
    )

    msg = cl.Message(content="", elements=source_elements)
    await msg.send()

    try:
        async with cl.Step(name=f"3️⃣  Generating answer", type="llm") as step:
            step.input = (
                f"Provider: `{provider}` · Model: `{model_id}`\n"
                f"Sending **{len(context.split())} words** of context "
                f"grounded on {len(chunks)} source chunks."
            )
            t0 = time.time()

            if provider == "claude":
                await stream_claude(model_id, context, query, msg)
            elif provider == "openai":
                await stream_openai(model_id, context, query, msg)
            elif provider == "azure":
                await stream_openai(model_id, context, query, msg, azure=True)

            gen_time = time.time() - t0
            step.output = f"Generated in {gen_time*1000:.0f}ms"

        await msg.stream_token(
            f"\n\n---\n_⚡ {embed_time*1000:.0f}ms embed · "
            f"{retrieval_time*1000:.0f}ms retrieve · "
            f"{gen_time*1000:.0f}ms generate · `{model_id}`_"
        )
        await msg.update()

    except Exception as e:
        await msg.stream_token(f"\n\n⚠️ Generation failed: {e}")
        await msg.update()
