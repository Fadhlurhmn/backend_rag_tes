"""
agents.py — Router + Manager + Specialist agent logic.
Session state disimpan in-memory per conversation_id.
"""

import os
from groq import Groq
from retriever import (
    stage1_gate,
    try_deterministic_calc,
    get_top_matches,
    decide_route,
    needs_context,
    passes_domain_floor,
    chunk_docs,
    ABSOLUTE_DENSE_FLOOR,
    MIN_SCORE_THRESHOLD,
)

# ─────────────────────────────────────────
# Groq client
# ─────────────────────────────────────────
client = Groq(api_key=os.environ["GROQ_API_KEY"])
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────
# Session state (in-memory, per conversation)
# ─────────────────────────────────────────
# Structure: { conversation_id: { active_doc, turns_since_active, history } }
_sessions: dict[str, dict] = {}

K_FINAL = 3
RELATIVE_CUTOFF = 0.25
MULTI_HOP_DETECT_CUTOFF = 0.45


def _get_session(conv_id: str) -> dict:
    if conv_id not in _sessions:
        _sessions[conv_id] = {
            "active_doc": None,
            "turns_since_active": 0,
            "history": [],
        }
    return _sessions[conv_id]


# ─────────────────────────────────────────
# Router
# ─────────────────────────────────────────
def route(pertanyaan: str, conv_id: str) -> tuple[str, list[str] | None]:
    """Return (route, chunks_or_None)."""
    sess = _get_session(conv_id)

    # Stage 1: chitchat gate
    if stage1_gate(pertanyaan) == "skip_to_manager":
        sess["history"].append(pertanyaan)
        return "manager", None

    # Deterministic leave calculator
    calc_answer = try_deterministic_calc(pertanyaan)
    if calc_answer is not None:
        sess["turns_since_active"] = 0
        sess["history"].append(pertanyaan)
        return "specialist", [calc_answer]

    # Hybrid retrieval — initial pass
    matches_awal = get_top_matches(pertanyaan, k=8)
    score_a = matches_awal[0][2]
    standalone_dense_raw = matches_awal[0][4]

    query_for_retrieval = pertanyaan
    chosen_matches = matches_awal

    # Context expansion for ambiguous / short queries
    if (
        standalone_dense_raw >= ABSOLUTE_DENSE_FLOOR
        and needs_context(pertanyaan, score_a)
        and sess["history"]
    ):
        context_text = " ".join(sess["history"][-2:]) + " " + pertanyaan
        matches_ctx = get_top_matches(context_text, k=8)
        if matches_ctx[0][2] > score_a:
            chosen_matches = matches_ctx
            query_for_retrieval = context_text

    # Decide initial route
    if decide_route(chosen_matches) != "specialist":
        sess["turns_since_active"] += 1
        sess["history"].append(pertanyaan)
        return "manager", None

    # Multi-hop detection
    docs_in_top5 = {m[1] for m in matches_awal if matches_awal[0][2] - m[2] <= MULTI_HOP_DETECT_CUTOFF}
    is_multi_hop = len(docs_in_top5) > 1

    candidate_idx: list[int] | None = None
    if is_multi_hop:
        k_use = max(K_FINAL, 2 * len(docs_in_top5))
    else:
        k_use = K_FINAL
        if sess["active_doc"] and sess["turns_since_active"] <= 2:
            candidate_idx = [i for i, d in enumerate(chunk_docs) if d == sess["active_doc"]]

    matches = get_top_matches(query_for_retrieval, k=k_use, candidate_idx=candidate_idx)

    # Fallback if narrowed search is weak
    if matches[0][2] < MIN_SCORE_THRESHOLD and candidate_idx is not None:
        matches = get_top_matches(query_for_retrieval, k=k_use, candidate_idx=None)

    if not passes_domain_floor(matches) or matches[0][2] < MIN_SCORE_THRESHOLD:
        sess["turns_since_active"] += 1
        sess["history"].append(pertanyaan)
        return "manager", None

    # Build filtered chunk list
    top_score = matches[0][2]
    if is_multi_hop:
        filtered: list[str] = []
        seen_docs: set[str] = set()
        for m in matches:
            if m[1] not in seen_docs or (top_score - m[2] <= RELATIVE_CUTOFF):
                filtered.append(m[0])
                seen_docs.add(m[1])
        filtered = filtered[: max(K_FINAL, len(docs_in_top5))]
    else:
        filtered = [m[0] for m in matches if top_score - m[2] <= RELATIVE_CUTOFF]

    sess["active_doc"] = matches[0][1] if not is_multi_hop else None
    sess["turns_since_active"] = 0
    sess["history"].append(pertanyaan)
    return "specialist", filtered


# ─────────────────────────────────────────
# Manager Agent
# ─────────────────────────────────────────
MANAGER_PROMPT = (
    "Kamu asisten HR internal. Jawab pertanyaan umum secara singkat dan ramah. "
    "Kalau pertanyaan menyangkut angka spesifik kebijakan (hari cuti, batas reimbursement, dll), "
    "katakan kamu akan mengecek dokumen dulu, jangan menebak. "
    "Gunakan Bahasa Indonesia."
)


def call_manager(pertanyaan: str) -> tuple[str, dict]:
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": MANAGER_PROMPT},
            {"role": "user", "content": pertanyaan},
        ],
    )
    usage = resp.usage
    log = {
        "role": "manager",
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }
    return resp.choices[0].message.content, log


# ─────────────────────────────────────────
# Specialist Agent
# ─────────────────────────────────────────
SPECIALIST_SYSTEM_TEMPLATE = (
    "Kamu adalah asisten HR yang sangat akurat. "
    "Jawab HANYA berdasarkan konteks-konteks berikut. "
    "Jika informasi tidak ada di konteks, jawab 'Informasi tidak tersedia di dokumen kebijakan.' "
    "Jangan mengarang. Gunakan Bahasa Indonesia.\n\n"
    "KONTEKS:\n{context}"
)


def call_specialist(pertanyaan: str, chunks: list[str]) -> tuple[str, dict]:
    # Jika chunk adalah hasil deterministic calc, langsung return tanpa LLM
    if len(chunks) == 1 and not chunks[0].startswith("Dokumen"):
        # Ini hasil deterministic calc
        return chunks[0], {"role": "specialist", "input_tokens": 0, "output_tokens": 0}

    context = "\n\n---\n\n".join(f"[Konteks {i+1}] {c}" for i, c in enumerate(chunks))
    system_prompt = SPECIALIST_SYSTEM_TEMPLATE.format(context=context)

    def _call():
        return client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pertanyaan},
            ],
        )

    resp = _call()
    content = resp.choices[0].message.content
    if not content or not content.strip():
        resp = _call()
        content = resp.choices[0].message.content

    if not content or not content.strip():
        content = "Maaf, terjadi kendala teknis saat memproses jawaban. Silakan coba tanyakan ulang."

    usage = resp.usage
    log = {
        "role": "specialist",
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }
    return content, log


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────
def ask(pertanyaan: str, conv_id: str) -> tuple[str, dict]:
    """
    Route question, call appropriate agent, return (answer, log_dict).
    log_dict: { role, input_tokens, output_tokens }
    """
    route_result, chunks = route(pertanyaan, conv_id)
    if route_result == "manager":
        answer, log = call_manager(pertanyaan)
    else:
        answer, log = call_specialist(pertanyaan, chunks)
    return answer, log
