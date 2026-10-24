"""Evaluate the four routers on the held-out test split."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from app.routers import embed_router, hybrid_router, laya_router, llm_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = PROJECT_ROOT / "data" / "eval_test.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent
TOOL_NAMES = [
    "check_patient_policy",
    "calculate_out_of_pocket",
    "request_prior_authorization",
    "escalate_to_human_specialist",
]
CONFUSION_LABELS = TOOL_NAMES + ["no_tool"]
ROUTERS = {
    "qwen": llm_router.route,
    "laya": laya_router.route,
    "embedding": embed_router.route,
    "hybrid": hybrid_router.route,
}
REPEATS = 3


def load_test_records(path=TEST_FILE):
    """Read and validate labeled rows from the test split only."""
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    required = {"id", "query", "tool", "category"}
    if not records:
        raise ValueError(f"The test split is empty: {path}")
    if any(set(record) != required for record in records):
        raise ValueError("Every test row must contain exactly id, query, tool, and category.")
    if any(record["tool"] not in TOOL_NAMES for record in records):
        raise ValueError("The test split contains an unknown expected tool label.")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("The test split contains duplicate ids.")
    return records


def _valid_tool(value):
    """Map invalid router outputs to a visible confusion-matrix label."""
    return value if value in TOOL_NAMES else "no_tool"


def _run_router(name, route, records):
    """Run each test query three times and exclude its first run as warmup."""
    results = []
    for record in records:
        for repeat in range(1, REPEATS + 1):
            prediction = route(record["query"])
            if repeat == 1:
                continue

            results.append(
                {
                    "router": name,
                    "id": record["id"],
                    "query": record["query"],
                    "category": record["category"],
                    "expected_tool": record["tool"],
                    "predicted_tool": _valid_tool(prediction.get("tool")),
                    "confidence": prediction.get("confidence", 0.0),
                    "latency_ms": prediction["latency_ms"],
                    "output_tokens": prediction["output_tokens"],
                    "repeat": repeat,
                }
            )
    return results


def _router_metrics(results):
    """Calculate routing accuracy, macro-F1, latency, token, and category metrics."""
    expected = [row["expected_tool"] for row in results]
    predicted = [row["predicted_tool"] for row in results]
    latencies = [row["latency_ms"] for row in results]
    tokens = [row["output_tokens"] for row in results]
    categories = defaultdict(list)

    for row in results:
        categories[row["category"]].append(
            row["predicted_tool"] == row["expected_tool"]
        )

    matrix = confusion_matrix(expected, predicted, labels=CONFUSION_LABELS)
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=TOOL_NAMES, average="macro", zero_division=0)
        ),
        "confusion_matrix_labels": CONFUSION_LABELS,
        "confusion_matrix": matrix.tolist(),
        "category_accuracy": {
            category: sum(values) / len(values)
            for category, values in sorted(categories.items())
        },
        "latency_ms": {
            "p50": float(statistics.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "output_tokens": {
            "total": int(sum(tokens)),
            "average_per_prediction": float(statistics.mean(tokens)),
        },
        "measured_predictions": len(results),
    }


def _write_outputs(summary, results):
    """Save the aggregate metrics as JSON and individual predictions as CSV."""
    json_path = OUTPUT_DIR / "results.json"
    csv_path = OUTPUT_DIR / "results.csv"
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        columns = [
            "router",
            "id",
            "query",
            "category",
            "expected_tool",
            "predicted_tool",
            "confidence",
            "latency_ms",
            "output_tokens",
            "repeat",
        ]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def run_evaluation():
    """Run all routers against the test split and persist benchmark results."""
    records = load_test_records()
    all_results = []
    metrics = {}

    for name, route in ROUTERS.items():
        router_results = _run_router(name, route, records)
        all_results.extend(router_results)
        metrics[name] = _router_metrics(router_results)

    summary = {
        "test_file": str(TEST_FILE.relative_to(PROJECT_ROOT)),
        "test_queries": len(records),
        "repeats_per_query": REPEATS,
        "warmup_runs_discarded_per_router": len(records),
        "measured_runs_per_query": REPEATS - 1,
        "routers": metrics,
    }
    json_path, csv_path = _write_outputs(summary, all_results)
    print(json.dumps(metrics, indent=2))
    print(f"Saved aggregate results to {json_path}")
    print(f"Saved per-query results to {csv_path}")
    return summary


if __name__ == "__main__":
    run_evaluation()
