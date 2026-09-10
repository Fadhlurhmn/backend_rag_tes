"""
database.py — Supabase client for saving and fetching chat messages.
"""

import os
from supabase import create_client, Client

_supabase: Client | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_ANON_KEY"]
        _supabase = create_client(url, key)
    return _supabase


def ensure_conversation(conv_id: str) -> str:
    """Pastikan conversation_id ada di DB, buat jika belum."""
    sb = get_supabase()
    res = sb.table("conversations").select("id").eq("id", conv_id).execute()
    if not res.data:
        sb.table("conversations").insert({"id": conv_id}).execute()
    return conv_id


def save_user_message(conv_id: str, question: str) -> str:
    sb = get_supabase()
    res = (
        sb.table("messages")
        .insert({
            "conversation_id": conv_id,
            "role": "user",
            "content": question,
        })
        .execute()
    )
    return res.data[0]["id"]


def save_agent_message(
    conv_id: str,
    answer: str,
    agent_role: str,
    input_tokens: int,
    output_tokens: int,
) -> str:
    sb = get_supabase()
    total = input_tokens + output_tokens
    res = (
        sb.table("messages")
        .insert({
            "conversation_id": conv_id,
            "role": "assistant",
            "content": answer,
            "agent_role": agent_role,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
        })
        .execute()
    )
    return res.data[0]["id"]


def get_history(conv_id: str) -> list[dict]:
    sb = get_supabase()
    res = (
        sb.table("messages")
        .select("*")
        .eq("conversation_id", conv_id)
        .order("created_at")
        .execute()
    )
    return res.data
