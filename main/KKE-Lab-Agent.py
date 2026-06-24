"""Enterprise RAG CLI — Native OpenAI Function Calling + FAISS + Ollama.

The model itself decides intent via its system prompt:
  - Greetings / general chat  → replies naturally (no tool call)
  - Document / FAQ questions  → calls search_knowledge_base or corpus_overview

Tools (OpenAI function-calling protocol, tool_choice="auto"):
  • search_knowledge_base — FAISS semantic search over ingested docs
  • corpus_overview        — doc count, titles, aggregate stats

Setup:
    ollama pull gpt-oss:20b && ollama pull nomic-embed-text
    pip install openai faiss-cpu numpy

Run:
    python enterprise_rag.py dataset.txt
"""
from __future__ import annotations
import sys, glob, asyncio, json
import numpy as np
import faiss
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
CHAT_MODEL      = "gpt-oss:20b"
EMBED_MODEL     = "nomic-embed-text"
EMBED_DIM       = 768
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 8
EMBED_BATCH     = 64
MAX_HISTORY     = 24
DOC_PREFIX      = "search_document: "
QUERY_PREFIX    = "search_query: "

client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# FAISS Vector Store
# ---------------------------------------------------------------------------
class VectorStore:
    def __init__(self) -> None:
        self.index   = faiss.IndexFlatIP(EMBED_DIM)
        self.chunks  : list[str] = []
        self.sources : list[str] = []
        self.titles  : list[str] = []

    @staticmethod
    def _parse_records(text: str) -> list[tuple[str, str]]:
        marker = "=== DOCUMENT START ==="
        blocks = text.split(marker) if marker in text else [text]
        out: list[tuple[str, str]] = []
        for b in blocks:
            b = b.replace("=== DOCUMENT END ===", "").strip()
            if not b:
                continue
            title = next((ln.split(":", 1)[1].strip() for ln in b.splitlines()
                          if ln.lower().startswith("title:")), "")
            body  = b.split("text:", 1)[-1].strip()
            out.append((title, body))
        return out

    @staticmethod
    def _chunk(title: str, body: str) -> list[str]:
        step = CHUNK_SIZE - CHUNK_OVERLAP
        tag  = f"[{title}]\n" if title else ""
        return [tag + body[i:i + CHUNK_SIZE].strip()
                for i in range(0, len(body), step)
                if body[i:i + CHUNK_SIZE].strip()]

    async def _embed(self, texts: list[str], prefix: str = "") -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = [prefix + t for t in texts[i:i + EMBED_BATCH]]
            resp  = await client.embeddings.create(model=EMBED_MODEL, input=batch)
            vecs.extend(d.embedding for d in resp.data)
        mat = np.asarray(vecs, dtype=np.float32)
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    async def add(self, source: str, text: str) -> int:
        chunks, srcs = [], []
        for title, body in self._parse_records(text):
            self.titles.append(title)
            for c in self._chunk(title, body):
                chunks.append(c)
                srcs.append(f"{source}{' :: ' + title if title else ''}")
        if not chunks:
            return 0
        vecs = await self._embed(chunks, DOC_PREFIX)
        self.index.add(vecs)
        self.chunks.extend(chunks)
        self.sources.extend(srcs)
        return len(chunks)

    async def search(self, query: str, k: int = TOP_K) -> str:
        if self.index.ntotal == 0:
            return "No documents indexed yet."
        q = await self._embed([query], QUERY_PREFIX)
        scores, ids = self.index.search(q, k)
        hits = []
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0])):
            if idx == -1:
                break
            hits.append(
                f"[{rank+1}] source={self.sources[idx]} (score={score:.3f})\n"
                f"{self.chunks[idx]}"
            )
        return "\n\n".join(hits) if hits else "No relevant passages found."

    def overview(self) -> str:
        titles = [t for t in self.titles if t]
        agg_keys = ("no. of", "record count", "years of excellence", "no.of")
        stats = sorted({
            ln.strip() for c in self.chunks for ln in c.splitlines()
            if ln.strip().lower().startswith(agg_keys) and any(d.isdigit() for d in ln)
        })
        lines = [f"Indexed documents: {len(titles)}"]
        if titles: lines.append("Document titles:\n- " + "\n- ".join(titles))
        if stats:  lines.append("Aggregate figures:\n- " + "\n- ".join(stats))
        return "\n\n".join(lines)


STORE = VectorStore()

# ---------------------------------------------------------------------------
# OpenAI Function-Calling Tool Schemas
# ---------------------------------------------------------------------------
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the indexed enterprise documents using semantic similarity. "
                "Call this when the user asks anything about the content of the "
                "uploaded documents — facts, details, summaries, comparisons, FAQs, "
                "or any domain-specific question that requires looking up information. "
                "Do NOT call this for greetings, small talk, or general knowledge "
                "questions that are unrelated to the documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural-language query capturing what the user wants to know."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "corpus_overview",
            "description": (
                "Returns high-level metadata about the indexed corpus: total document "
                "count, all document titles, and any aggregate numeric figures present. "
                "Call this when the user asks how many documents are loaded, wants a "
                "list of all documents, or asks broad overview / counting questions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt — model decides intent, no hardcoding
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly and intelligent enterprise assistant.

You have two tools available:
- search_knowledge_base: searches the user's uploaded documents
- corpus_overview: lists what documents are loaded and their stats

HOW TO DECIDE WHAT TO DO:
1. If the user sends a greeting, small talk, or a general conversational message
   (e.g. "hi", "hello", "how are you", "thanks", "bye") — respond warmly and
   naturally WITHOUT calling any tool.

2. If the user asks a question about the content of the uploaded documents, an
   FAQ, or anything that requires looking up specific information — call
   search_knowledge_base with a well-formed query.

3. If the user asks how many documents are loaded, what documents exist, or wants
   an overview — call corpus_overview.

4. If the intent is ambiguous, prefer calling search_knowledge_base to attempt
   a retrieval before concluding you don't know.

ANSWER RULES:
- For document questions: answer ONLY from tool results. Cite the source.
- For general chat: reply naturally and helpfully.
- Never say "I don't have that information" for a greeting or chitchat.
- If a tool returns no useful result, say so honestly and suggest the user
  rephrase or ask something else."""

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
async def dispatch(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return await STORE.search(args.get("query", ""))
    if name == "corpus_overview":
        return STORE.overview()
    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# Agentic loop — handles multi-turn tool calls until model gives final answer
# ---------------------------------------------------------------------------
async def run_agent(history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    while True:
        resp = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",   # model decides: reply or call a tool
            temperature=0.2,
        )
        msg = resp.choices[0].message

        # No tool calls → model decided to reply directly (chat or final answer)
        if not msg.tool_calls:
            return msg.content or ""

        # Append the assistant's tool-call turn to messages
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        # Execute all tool calls in parallel
        results = await asyncio.gather(*[
            dispatch(tc.function.name, json.loads(tc.function.arguments))
            for tc in msg.tool_calls
        ])

        # Feed each result back as a tool message
        for tc, result in zip(msg.tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })
        # Loop → model now sees tool results and either answers or calls again

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
async def ingest(patterns: list[str]) -> None:
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    if not paths:
        sys.exit("No files matched. Example: python enterprise_rag.py dataset.txt")
    print("Indexing documents ...")
    for path in paths:
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError as err:
            print(f"  skip {path}: {err}")
            continue
        n = await STORE.add(path, text)
        print(f"  {path}: {n} chunks  (FAISS total: {STORE.index.ntotal})")

# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------
async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    await ingest(sys.argv[1:])
    print(f"\nReady — {STORE.index.ntotal} vectors indexed. Type 'exit' to quit.\n")

    history: list[dict] = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": question})
        try:
            answer = await run_agent(history)
        except Exception as err:
            print(f"bot> error: {err}\n")
            history.pop()
            continue

        print(f"\nbot> {answer}\n")
        history.append({"role": "assistant", "content": answer})
        history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    asyncio.run(main())