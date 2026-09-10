"""
retriever.py — Hybrid Dense (BGE-M3) + BM25 retrieval engine.
Ported 1:1 dari prototype notebook (versi manual).
File-file data di-load sekali saat startup FastAPI.
"""

import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

# ─────────────────────────────────────────
# Thresholds (sama persis dengan notebook)
# ─────────────────────────────────────────
ABSOLUTE_DENSE_FLOOR = 0.48
BM25_SIGNAL_FLOOR = 0.5
MIN_SCORE_THRESHOLD = 0.30
MARGIN_THRESHOLD = 0.05
CONFIDENT_THRESHOLD = 0.7

CHITCHAT_PATTERNS = [
    "halo", "hai", "hi", "pagi", "siang", "sore", "malam",
    "terima kasih", "makasih", "oke", "baik", "sip",
    "kamu siapa", "apa kabar", "test", "coba",
]

# ─────────────────────────────────────────
# Load assets (sekali saat module diimport)
# ─────────────────────────────────────────
print("[retriever] Loading embedding model…")
embed_model = SentenceTransformer("BAAI/bge-m3")

print("[retriever] Loading data assets…")
with open(DATA_DIR / "all_chunks.json", "r", encoding="utf-8") as f:
    all_chunks: list[list[str]] = json.load(f)

chunk_docs: list[str] = [c[0] for c in all_chunks]
chunk_texts: list[str] = [c[1] for c in all_chunks]

chunk_matrix: np.ndarray = np.load(DATA_DIR / "chunk_matrix.npy")

with open(DATA_DIR / "bm25_index.pkl", "rb") as f:
    bm25_index = pickle.load(f)

print(f"[retriever] Ready — {len(chunk_texts)} chunks, vector dim {chunk_matrix.shape[1]}")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ─────────────────────────────────────────
# Core retrieval
# ─────────────────────────────────────────
def get_top_matches(
    query: str,
    k: int = 1,
    candidate_idx: list[int] | None = None,
    alpha: float = 0.5,
) -> list[tuple]:
    """Return list of (text, doc_label, combined_score, idx, dense_score)."""
    query_vec = embed_model.encode(query, normalize_embeddings=True)
    dense_scores: np.ndarray = chunk_matrix @ query_vec

    bm25_scores = np.array(bm25_index.get_scores(simple_tokenize(query)))
    bm25_raw_max = bm25_scores.max() if len(bm25_scores) else 0.0
    bm25_norm = np.zeros_like(bm25_scores) if bm25_raw_max < BM25_SIGNAL_FLOOR else _normalize(bm25_scores)

    combined = alpha * _normalize(dense_scores) + (1 - alpha) * bm25_norm

    idx_pool = candidate_idx if candidate_idx is not None else range(len(chunk_texts))
    ranked = sorted(idx_pool, key=lambda i: combined[i], reverse=True)[:k]
    return [(chunk_texts[i], chunk_docs[i], combined[i], i, dense_scores[i]) for i in ranked]


def passes_domain_floor(matches: list[tuple]) -> bool:
    return matches[0][4] >= ABSOLUTE_DENSE_FLOOR


def decide_route(matches: list[tuple]) -> str:
    if not passes_domain_floor(matches):
        return "manager"
    scores = [m[2] for m in matches]
    top_score = scores[0]
    baseline = np.mean(scores[1:]) if len(scores) > 1 else 0.0
    margin = top_score - baseline
    if top_score < MIN_SCORE_THRESHOLD:
        return "manager"
    if margin > MARGIN_THRESHOLD:
        return "specialist"
    return "manager"


def needs_context(pertanyaan: str, score_alone: float) -> bool:
    if score_alone >= CONFIDENT_THRESHOLD:
        return False
    short = len(pertanyaan.split()) <= 6
    has_referent = any(
        w in pertanyaan.lower() for w in ["itu", "nya", "tersebut", "bagaimana", "gimana"]
    )
    ambiguous = 0.3 <= score_alone < 0.6
    return (short and has_referent) or ambiguous


# ─────────────────────────────────────────
# Stage 1 gate (chitchat)
# ─────────────────────────────────────────
def stage1_gate(pertanyaan: str) -> str:
    p = pertanyaan.lower().strip()
    if len(p.split()) <= 3 and any(pat in p for pat in CHITCHAT_PATTERNS):
        return "skip_to_manager"
    return "proceed_to_stage2"


# ─────────────────────────────────────────
# Deterministic leave calculator
# ─────────────────────────────────────────
CUTI_TAHUNAN_HARI = 12
BULAN_PER_TAHUN = 12
MIN_BULAN_UNTUK_CUTI = 3


def try_deterministic_calc(pertanyaan: str) -> str | None:
    p = pertanyaan.lower()
    if "cuti" not in p:
        return None
    if any(kw in p for kw in ["menikah", "duka", "melahirkan", "unpaid", "tanpa gaji"]):
        return None
    m = re.search(r"(\d+)\s*bulan", p)
    if not m:
        return None
    if not any(kw in p for kw in ["berapa", "jatah", "hak", "dapat", "punya"]):
        return None

    bulan_kerja = int(m.group(1))
    if bulan_kerja < MIN_BULAN_UNTUK_CUTI:
        return (
            f"Dengan masa kerja {bulan_kerja} bulan, Anda belum berhak atas cuti tahunan "
            f"(minimal 3 bulan masa kerja). Anda hanya berhak cuti sakit dengan surat dokter."
        )
    if bulan_kerja >= 12:
        hari = CUTI_TAHUNAN_HARI
        keterangan = "cuti tahunan penuh (masa kerja sudah ≥ 12 bulan)"
    else:
        hari = math.floor(bulan_kerja / BULAN_PER_TAHUN * CUTI_TAHUNAN_HARI)
        keterangan = f"cuti prorata ({bulan_kerja} bulan kerja / 12 × {CUTI_TAHUNAN_HARI} hari, dibulatkan ke bawah)"
    return f"Dengan masa kerja {bulan_kerja} bulan, Anda berhak atas {hari} hari {keterangan}."
