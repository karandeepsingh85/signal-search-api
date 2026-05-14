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
    """Strip surrounding quote marks that appear on some ad hoc article titles."""
    return title.strip().strip('"').strip('“”')


def _short_description(article: dict) -> str:
    """
    Returns the best short description available for the article.
    Prefers why_exist (editorial framing), falls back to article_angle.
    Returns empty string if the text is too short or looks like a raw YAML marker.
    """
    text = article.get("why_exist") or article.get("article_angle") or ""
    # Skip YAML block scalar markers and other short non-sentences
    text = text.strip().lstrip(">").strip()
    if not text or len(text) < 20:
        return ""
    first_sentence = text.split(".")[0].strip()
    if len(first_sentence) < 15:
        return ""
    if len(first_sentence) > 160:
        return first_sentence[:157] + "..."
    return first_sentence + "."
