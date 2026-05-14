"""
search_api/app.py

Semantic search API for Signal for Men.

Loads the article search index on startup, embeds incoming reader queries
using the same model used to build the index (all-MiniLM-L6-v2), and returns
ranked article results by cosine similarity.

Endpoints:
    GET /search?q=your+query&top=8
    GET /health

Deploy to Render: connect this folder as a Python web service.
Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Startup — load index and model once
# ---------------------------------------------------------------------------

INDEX_PATH = Path(__file__).parent / "article_search_index.json"

with open(INDEX_PATH, encoding="utf-8") as f:
    _index = json.load(f)

# Pre-stack all embeddings into a matrix for fast dot-product scoring
_embeddings = np.array([a["embedding"] for a in _index], dtype=np.float32)

# Load the same model used to build the index — queries must use identical model
_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Signal for Men — Search API")

# Allow requests from the Ghost site and localhost (for development)
_allowed_origins = [
    "https://signalformen.com",
    "https://www.signalformen.com",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

FAMILY_LABELS = {
    "signal":  "Signal",
    "tools":   "Tools",
    "reframe": "Reframe",
}


@app.get("/")
def root():
    """Root endpoint — confirms the service is alive."""
    return {"ok": True, "articles": len(_index), "model": "all-MiniLM-L6-v2"}


@app.get("/health")
def health():
    """Health check — same as root, explicit path for Render health checks."""
    return {"ok": True, "articles": len(_index), "model": "all-MiniLM-L6-v2"}


@app.get("/search")
def search(
    q: str = Query(..., min_length=2, description="Reader's search query"),
    top: int = Query(8, ge=1, le=20, description="Number of results to return"),
):
    """
    Embed the reader query and return the top N articles by cosine similarity.

    Returns articles above a minimum relevance threshold (0.10) only.
    Results include title, family, URL, a short description, and similarity score.
    """
    # Embed the query with the same model and normalisation used for the index
    query_vector = _model.encode(q, normalize_embeddings=True)

    # Cosine similarity = dot product of unit-normalised vectors
    scores = _embeddings @ query_vector

    # Sort descending, take top N above threshold
    threshold = 0.10
    ranked = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )

    results = []
    for idx, score in ranked:
        if float(score) < threshold:
            break
        if len(results) >= top:
            break

        article = _index[idx]
        results.append({
            "title":       _clean_title(article.get("title") or article.get("seo_title") or ""),
            "family":      FAMILY_LABELS.get(article.get("family", ""), article.get("family", "")),
            "url":         article.get("ghost_url", ""),
            "score":       round(float(score), 3),
            "description": _short_description(article),
        })

    return {"query": q, "results": results}


def _clean_title(title: str) -> str:
    “””Strip surrounding quote marks that appear on some ad hoc article titles.”””
    return title.strip().strip('”').strip('“”')


# Phrases that indicate internal editorial/planning text — not for readers.
# Note: “ research” alone is too broad (hits “Gottman research”); use the specific
# editorial state names instead.
_INTERNAL_PHRASES = [
    “(score”,                   # research score refs: “(score 35)”
    “score “,                   # “Score 35–36” at the start
    “inventory”,                # “The inventory covers…” / “existing inventory”
    “crisis research”,          # editorial research-state labels
    “philosophical research”,
    “optimisation research”,
    “optimization research”,
    “distinct practical need”,
    “rubric”,
    “highest-scoring”,
    “low-confidence”,
    “seo entry”,
    “diagnostic seo”,
    “state 2 q”,                # research question refs: “State 2 Q47”
    “state 4 q”,
    “state 5 q”,
    “this piece”,               # editorial voice: “This piece explores…”
    “existing content”,
    “existing pieces”,
    “the platform”,             # “on the platform” — internal gap analysis
    “no content on”,
    “generation mode”,
    “ghost admin”,
    “claude”,
    “practical companion to”,   # cross-ref to another article ID
    “series closer”,            # editorial label for the closing article
]


def _is_internal(text: str) -> bool:
    “””Returns True if the first sentence contains internal planning/editorial language.”””
    t = text.lower()
    return any(phrase in t for phrase in _INTERNAL_PHRASES)


def _short_description(article: dict) -> str:
    “””
    Returns a reader-facing description for a search result.

    Uses article_angle only — why_exist is editorial/planning metadata and is
    never shown to readers. Returns empty string if the text is internal, too
    short, or contains planning language.

    meta_description/custom_excerpt are not yet in the index. When added,
    prefer them first (they are written for readers, not editors).
    “””
    text = (article.get(“article_angle”) or “”).strip().lstrip(“>”).strip()
    if not text or len(text) < 20:
        return “”
    # Check only the first sentence — later sentences may be internal even
    # when the opener is clean (e.g. “Most men know X. This piece gives you…”)
    first_sentence = text.split(“.”)[0].strip()
    if len(first_sentence) < 15:
        return “”
    if _is_internal(first_sentence):
        return “”
    if len(first_sentence) > 160:
        return first_sentence[:157] + “...”
    return first_sentence + “.”
