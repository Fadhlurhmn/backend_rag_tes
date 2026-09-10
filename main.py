"""
main.py — FastAPI entry point.
"""

import uuid
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse, HistoryResponse, HistoryMessage
from agents import ask
from database import (
    ensure_conversation,
    save_user_message,
    save_agent_message,
    get_history,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # retriever.py loads assets at import time — nothing extra needed
    yield


app = FastAPI(
    title="BrandnPurpose HR Chat API",
    version="1.0.0",
    lifespan=lifespan,
)

# ─────────────────────────────────────────
# CORS — izinkan Next.js frontend
# ─────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://*.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ganti ke ALLOWED_ORIGINS saat production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "BrandnPurpose HR Chat API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Resolve atau buat conversation_id
    conv_id = req.conversation_id or str(uuid.uuid4())

    try:
        ensure_conversation(conv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # Simpan pesan user
    try:
        save_user_message(conv_id, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB save user error: {e}")

    # Jalankan agent
    try:
        answer, log = ask(req.question, conv_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    agent_role = log["role"]
    input_tokens = log["input_tokens"]
    output_tokens = log["output_tokens"]
    total_tokens = input_tokens + output_tokens

    # Simpan jawaban agent
    try:
        msg_id = save_agent_message(
            conv_id, answer, agent_role, input_tokens, output_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB save agent error: {e}")

    return ChatResponse(
        answer=answer,
        agent=agent_role,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        conversation_id=conv_id,
        message_id=msg_id,
    )


@app.get("/history/{conversation_id}", response_model=HistoryResponse)
async def history(conversation_id: str):
    try:
        rows = get_history(conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    messages = [
        HistoryMessage(
            id=r["id"],
            role=r["role"],
            content=r["content"],
            agent_role=r.get("agent_role"),
            input_tokens=r.get("input_tokens"),
            output_tokens=r.get("output_tokens"),
            total_tokens=r.get("total_tokens"),
            created_at=str(r["created_at"]),
        )
        for r in rows
    ]
    return HistoryResponse(conversation_id=conversation_id, messages=messages)
