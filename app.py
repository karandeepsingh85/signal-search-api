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
import re
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

    The query is normalised before embedding (typo correction, apostrophe
    normalisation) but the original query is returned in the response.

    Threshold logic:
      - Primary threshold 0.18: aim to return only strong matches.
      - If fewer than 3 results pass 0.18, fall back to a floor of 0.14
        and return the best available up to 3.
      - Nothing below 0.14 is ever returned.
    """
    embed_q = normalise_query(q)
    query_vector = _model.encode(embed_q, normalize_embeddings=True)

    scores = _embeddings @ query_vector
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Collect everything above the floor threshold
    _PRIMARY = 0.18
    _FLOOR   = 0.14

    candidates = []
    for i, score in ranked:
        if float(score) < _FLOOR:
            break
        candidates.append((i, float(score)))

    # Use primary threshold; guarantee at least 3 results if floor allows
    primary = [(i, s) for i, s in candidates if s >= _PRIMARY]
    selected = primary[:top] if len(primary) >= 3 else candidates[:3]

    results = []
    for i, score in selected:
        article = _index[i]
        results.append({
            "title":       _clean_title(article.get("title") or article.get("seo_title") or ""),
            "family":      FAMILY_LABELS.get(article.get("family", ""), article.get("family", "")),
            "url":         article.get("ghost_url", ""),
            "score":       round(score, 3),
            "description": _short_description(article),
        })

    return {"query": q, "results": results}


def _clean_title(title: str) -> str:
    """Strip surrounding quote marks that appear on some ad hoc article titles."""
    return title.strip().strip('"').strip('""')


# ---------------------------------------------------------------------------
# Query normalisation
# ---------------------------------------------------------------------------

# Multi-word phrase corrections applied before word-level corrections.
# Longer/more-specific patterns listed first.
_PHRASE_CORRECTIONS = [
    ("my wife an i",      "my wife and i"),
    ("my partner an i",   "my partner and i"),
    ("me an my wife",     "me and my wife"),
    ("me an my partner",  "me and my partner"),
    (" an i keep",        " and i keep"),
    (" an i ",            " and i "),
]

# Word-level typo corrections keyed by the bad spelling.
_WORD_CORRECTIONS = {
    "fiting":       "fighting",
    "fightng":      "fighting",
    "figthing":     "fighting",
    "realtionship": "relationship",
    "relationsip":  "relationship",
    "relatonship":  "relationship",
    "wokr":         "work",
    "freinds":      "friends",
    "lonley":       "lonely",
    "numbb":        "numb",
    "cant":         "can't",
    "im":           "i'm",
    "ive":          "i've",
}

# Pre-compile word-boundary patterns once at import time
_WORD_RE = {
    word: re.compile(r"\b" + re.escape(word) + r"\b")
    for word in _WORD_CORRECTIONS
}


def normalise_query(q: str) -> str:
    """
    Lightweight pre-embedding normalisation for reader queries.

    Fixes specific known typos and phrase variants that confuse the embedding
    model (e.g. "fiting" matches "fitting" clothes rather than "fighting").
    Not a full spellchecker — only corrects patterns we know cause bad results.

    The original query is returned in the API response; only the normalised
    form is passed to the embedding model.
    """
    # Normalise whitespace
    q = re.sub(r"\s+", " ", q).strip()
    # Curly apostrophes → straight
    q = q.replace("'", "'").replace("'", "'")
    q = q.replace(""", '"').replace(""", '"')

    norm = q.lower()

    # Phrase-level corrections first (most specific)
    for wrong, right in _PHRASE_CORRECTIONS:
        norm = norm.replace(wrong, right)

    # Word-level corrections (whole-word match only)
    for word, correction in _WORD_CORRECTIONS.items():
        norm = _WORD_RE[word].sub(correction, norm)

    return re.sub(r"\s+", " ", norm).strip()


# ---------------------------------------------------------------------------
# Description suppression
# ---------------------------------------------------------------------------

# Phrases that indicate internal editorial/planning text — not for readers.
# "research" alone is too broad (hits "Gottman research" etc.); use the specific
# editorial state-name labels instead.
_INTERNAL_PHRASES = [
    "(score",                   # score refs: "(score 35)"
    "score ",                   # "Score 35–36"
    "inventory",                # "The inventory covers…"
    "crisis research",          # editorial research-state labels
    "philosophical research",
    "optimisation research",
    "optimization research",
    "optimisation state",       # "Optimisation state men want…"
    "distinct practical need",
    "rubric",
    "highest-scoring",
    "low-confidence",
    "seo entry",
    "diagnostic seo",
    "state 2 q",                # research question refs: "State 2 Q47"
    "state 4 q",
    "state 5 q",
    "state q",                  # catch any "State N Q…" variant
    "this piece",               # editorial: "This piece explores…"
    "existing content",
    "existing pieces",
    "the platform",             # "on the platform" — internal gap analysis
    "no content on",
    "generation mode",
    "ghost admin",
    "claude",
    "practical companion to",   # cross-ref: "Practical companion to 4.3-A"
    "series closer",            # editorial label for closing article
    "covers male",              # "inventory covers male anger…"
]


def _is_internal(text: str) -> bool:
    """Returns True if the first sentence contains internal planning/editorial language."""
    t = text.lower()
    return any(phrase in t for phrase in _INTERNAL_PHRASES)


def _short_description(article: dict) -> str:
    """
    Returns a reader-facing description for a search result.

    Uses article_angle only — why_exist is editorial/planning metadata and is
    never shown to readers. Returns empty string if the text is internal, too
    short, or contains planning language.

    meta_description/custom_excerpt are not yet in the index. When added,
    prefer them first (they are written for readers, not editors).
    """
    text = (article.get("article_angle") or "").strip().lstrip(">").strip()
    if not text or len(text) < 20:
        return ""
    # Check only the first sentence — later sentences may be internal even
    # when the opener is clean (e.g. "Most men know X. This piece gives you…")
    first_sentence = text.split(".")[0].strip()
    if len(first_sentence) < 15:
        return ""
    if _is_internal(first_sentence):
        return ""
    if len(first_sentence) > 160:
        return first_sentence[:157] + "..."
    return first_sentence + "."
