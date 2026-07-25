import sqlite3
import json
import math
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Local Chatbot", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DB_PATH = "chatbot.db"

# Known embedding models to exclude from chat dropdown
EMBEDDING_MODELS = {
    "nomic-embed-text",
    "all-minilm",
    "mxbai-embed",
    "snowflake-arctic-embed",
}


# ========== DATABASE ==========
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT DEFAULT 'New Chat',
            system_prompt TEXT DEFAULT 'You are a helpful assistant.',
            temperature REAL DEFAULT 0.7,
            top_p REAL DEFAULT 0.9,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            filename TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            content TEXT,
            embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ========== HELPERS ==========
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=60.0,
        )
        return resp.json()["embedding"]


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


SIMILARITY_THRESHOLD = 0.35

# Meta-questions that should NEVER trigger RAG
META_PATTERNS = [
    "did i upload",
    "have i uploaded",
    "what did i upload",
    "did you see",
    "do you see",
    "what files",
    "any files",
    "what did i say",
    "what did i ask",
    "what was my",
    "did i send",
    "do i have",
    "is there a file",
    "upload a pdf",
    "upload a file",
    "any documents",
]


def is_meta_question(query: str) -> bool:
    q = query.lower().strip()
    return any(p in q for p in META_PATTERNS)


async def retrieve_context(conv_id: int, query: str, top_k: int = 4) -> str:
    if is_meta_question(query):
        return ""

    try:
        query_emb = await get_embedding(query)
    except Exception:
        return ""

    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT COUNT(*) as doc_count FROM documents WHERE conversation_id = ?",
        (conv_id,),
    )
    doc_count = c.fetchone()["doc_count"]

    c.execute(
        """
        SELECT ch.content, ch.embedding
        FROM chunks ch
        JOIN documents d ON ch.document_id = d.id
        WHERE d.conversation_id = ?
    """,
        (conv_id,),
    )
    rows = c.fetchall()
    conn.close()

    scored = []
    for row in rows:
        emb = json.loads(row["embedding"])
        score = cosine_similarity(query_emb, emb)
        if score >= SIMILARITY_THRESHOLD:
            scored.append((score, row["content"]))

    scored.sort(reverse=True)
    top = scored[:top_k]

    if not top:
        if doc_count > 0:
            return f"\n\nThe user has uploaded {doc_count} document(s). However, their current question is too general to retrieve specific excerpts. If they are asking broadly about the document, encourage them to ask a more specific question (e.g., about skills, experience, or a specific topic)."
        return ""

    context = "\n\n".join(
        [f"[Excerpt {i + 1}]\n{content}" for i, (_, content) in enumerate(top)]
    )
    return f"\n\nThe user has uploaded documents. Below are the most relevant excerpts. ONLY reference this information if the user is explicitly asking about the document content. Do NOT mention these excerpts for general questions, greetings, or questions about the conversation itself.\n\n{context}"


# ========== MODELS ==========
class ChatMessage(BaseModel):
    message: str
    model: str = DEFAULT_MODEL
    conversation_id: Optional[int] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


class OpenAIChatRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[dict]
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 0.9


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None


# ========== ROUTES ==========
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                return {"status": "ok", "ollama": "connected"}
    except Exception:
        pass
    return {"status": "error", "ollama": "disconnected"}


@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
            data = resp.json()
            if "models" in data:
                data["models"] = [
                    m
                    for m in data["models"]
                    if not any(
                        embed in m.get("name", "").lower() for embed in EMBEDDING_MODELS
                    )
                ]
            return data
    except Exception as e:
        return {"error": str(e), "models": []}


@app.get("/api/conversations")
async def get_conversations():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return {"conversations": [dict(r) for r in rows]}


@app.post("/api/conversations")
async def create_conversation():
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO conversations DEFAULT VALUES")
    conn.commit()
    conv_id = c.lastrowid
    c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    row = c.fetchone()
    conn.close()
    return dict(row)


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    conv = c.fetchone()
    c.execute(
        "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,),
    )
    msgs = c.fetchall()
    c.execute(
        "SELECT id, filename, created_at FROM documents WHERE conversation_id = ?",
        (conv_id,),
    )
    docs = c.fetchall()
    conn.close()
    return {
        "conversation": dict(conv) if conv else None,
        "messages": [dict(m) for m in msgs],
        "documents": [dict(d) for d in docs],
    }


@app.patch("/api/conversations/{conv_id}")
async def update_conversation(conv_id: int, payload: ConversationUpdate):
    conn = get_db()
    c = conn.cursor()
    fields = []
    values = []
    for k, v in payload.model_dump(exclude_unset=True).items():
        fields.append(f"{k} = ?")
        values.append(v)
    if fields:
        values.append(conv_id)
        c.execute(
            f"UPDATE conversations SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        conn.commit()
    conn.close()
    return {"ok": True}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "DELETE FROM chunks WHERE document_id IN (SELECT id FROM documents WHERE conversation_id = ?)",
        (conv_id,),
    )
    c.execute("DELETE FROM documents WHERE conversation_id = ?", (conv_id,))
    c.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/documents")
async def get_documents(conv_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT id, filename, created_at FROM documents WHERE conversation_id = ?",
        (conv_id,),
    )
    rows = c.fetchall()
    conn.close()
    return {"documents": [dict(r) for r in rows]}


@app.post("/api/conversations/{conv_id}/documents")
async def upload_document(conv_id: int, file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    if file.filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            from io import BytesIO

            reader = PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        except ImportError:
            text = "[PDF parsing requires: uv pip install pypdf]"

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO documents (conversation_id, filename, content) VALUES (?, ?, ?)",
        (conv_id, file.filename, text),
    )
    doc_id = c.lastrowid

    chunks = chunk_text(text)
    for chunk in chunks:
        try:
            embedding = await get_embedding(chunk)
            c.execute(
                "INSERT INTO chunks (document_id, content, embedding) VALUES (?, ?, ?)",
                (doc_id, chunk, json.dumps(embedding)),
            )
        except Exception as e:
            print(f"Embedding error: {e}")

    conn.commit()
    conn.close()
    return {"ok": True, "document_id": doc_id, "chunks": len(chunks)}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/api/chat")
async def chat_stream(payload: ChatMessage):
    conv_id = payload.conversation_id

    conn = get_db()
    c = conn.cursor()
    if not conv_id:
        c.execute("INSERT INTO conversations DEFAULT VALUES")
        conn.commit()
        conv_id = c.lastrowid

    c.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
    row = c.fetchone()
    conv = dict(row) if row else None
    conn.close()

    system_prompt = payload.system_prompt or (
        conv["system_prompt"] if conv else "You are a helpful assistant."
    )
    temperature = (
        payload.temperature
        if payload.temperature is not None
        else (conv["temperature"] if conv else 0.7)
    )
    top_p = (
        payload.top_p if payload.top_p is not None else (conv["top_p"] if conv else 0.9)
    )

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT title FROM conversations WHERE id = ?", (conv_id,))
    title_row = c.fetchone()
    if title_row and title_row["title"] == "New Chat":
        new_title = (
            payload.message[:60] + "..."
            if len(payload.message) > 60
            else payload.message
        )
        c.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (new_title, conv_id)
        )
        conn.commit()
    conn.close()

    rag_context = ""
    if conv_id:
        rag_context = await retrieve_context(conv_id, payload.message)

    messages = [{"role": "system", "content": system_prompt + rag_context}]
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,),
    )
    hist = c.fetchall()
    conn.close()
    for h in hist[-12:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": payload.message})

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conv_id, "user", payload.message),
    )
    conn.commit()
    conn.close()

    ollama_payload = {
        "model": payload.model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature, "top_p": top_p},
    }

    async def generate():
        bot_content = ""
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload, timeout=180.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    chunk = data["message"]["content"]
                                    bot_content += chunk
                                    yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                                if data.get("done"):
                                    yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        finally:
            if bot_content:
                conn = get_db()
                c = conn.cursor()
                c.execute(
                    "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    (conv_id, "assistant", bot_content),
                )
                c.execute(
                    "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (conv_id,),
                )
                conn.commit()
                conn.close()

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def openai_chat(request: OpenAIChatRequest):
    ollama_payload = {
        "model": request.model,
        "messages": request.messages,
        "stream": request.stream,
        "options": {"temperature": request.temperature, "top_p": request.top_p},
    }

    if request.stream:

        async def generate():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", f"{OLLAMA_URL}/api/chat", json=ollama_payload, timeout=180.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    chunk = data["message"]["content"]
                                    sse_data = {
                                        "id": "chatcmpl-local",
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp()),
                                        "model": request.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": chunk},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(sse_data)}\n\n"
                                if data.get("done"):
                                    done_data = {
                                        "id": "chatcmpl-local",
                                        "object": "chat.completion.chunk",
                                        "created": int(datetime.now().timestamp()),
                                        "model": request.model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": "stop",
                                            }
                                        ],
                                    }
                                    yield f"data: {json.dumps(done_data)}\n\n"
                                    yield "data: [DONE]\n\n"
                            except json.JSONDecodeError:
                                continue

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={**ollama_payload, "stream": False},
                timeout=180.0,
            )
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return {
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
