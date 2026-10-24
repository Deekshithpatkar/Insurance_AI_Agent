"""Sentence-embedding and logistic-regression tool router."""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_FILE = Path(__file__).resolve().parents[2] / "data" / "eval_train.jsonl"
SEED = 2026

_model = None
_classifier = None


def _load_router():
    """Train the classifier on the training split and cache it for later queries."""
    global _model, _classifier
    if _model is None or _classifier is None:
        with TRAIN_FILE.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]

        _model = SentenceTransformer(MODEL_NAME)
        embeddings = _model.encode(
            [record["query"] for record in records],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        _classifier = LogisticRegression(max_iter=1000, random_state=SEED)
        _classifier.fit(embeddings, [record["tool"] for record in records])
    return _model, _classifier


def route(query: str) -> dict:
    """Classify a query using embeddings and return its confidence and latency."""
    start = time.perf_counter()
    model, classifier = _load_router()
    embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    probabilities = classifier.predict_proba(embedding)[0]
    choice_index = probabilities.argmax()
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "tool": str(classifier.classes_[choice_index]),
        "confidence": float(probabilities[choice_index]),
        "latency_ms": round(latency_ms, 2),
        "output_tokens": 0,
    }
