"""Select the hybrid confidence threshold from TRAIN data and plot the trade-off."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from app.routers import hybrid_router, laya_router, llm_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_ROOT / "data" / "eval_train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent
THRESHOLDS = tuple(value / 100 for value in range(30, 96, 5))
VALID_TOOLS = hybrid_router.VALID_TOOLS


def load_train_records(path=TRAIN_FILE):
    """Read labeled rows from the training split only."""
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    required = {"id", "query", "tool", "category"}
    if not records:
        raise ValueError(f"The training split is empty: {path}")
    if any(set(record) != required for record in records):
        raise ValueError("Every training row must contain exactly id, query, tool, and category.")
    if any(record["tool"] not in VALID_TOOLS for record in records):
        raise ValueError("The training split contains an unknown expected tool label.")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("The training split contains duplicate ids.")
    return records


def collect_predictions(records):
    """Cache Laya and required Qwen fallback predictions for TRAIN queries."""
    cached = []
    max_threshold = max(THRESHOLDS)
    for record in records:
        laya_result = laya_router.route(record["query"])
        valid_laya_tool = laya_result["tool"] in VALID_TOOLS
        needs_fallback = (
            not valid_laya_tool or laya_result["confidence"] < max_threshold
        )
        fallback_result = (
            llm_router.route(record["query"]) if needs_fallback else None
        )
        cached.append(
            {
                "id": record["id"],
                "query": record["query"],
                "expected_tool": record["tool"],
                "laya_tool": laya_result["tool"],
                "laya_confidence": laya_result["confidence"],
                "fallback_tool": (
                    fallback_result["tool"] if fallback_result is not None else None
                ),
            }
        )
    return cached


def sweep_thresholds(cached_predictions):
    """Measure accuracy and Laya-only traffic for each candidate threshold."""
    if not cached_predictions:
        raise ValueError("Cannot sweep thresholds without TRAIN predictions.")

    results = []
    total_queries = len(cached_predictions)
    for threshold in THRESHOLDS:
        correct = 0
        laya_handled = 0
        for prediction in cached_predictions:
            use_laya = (
                prediction["laya_tool"] in VALID_TOOLS
                and prediction["laya_confidence"] >= threshold
            )
            if use_laya:
                laya_handled += 1
                chosen_tool = prediction["laya_tool"]
            else:
                chosen_tool = prediction["fallback_tool"]
                if chosen_tool is None:
                    raise ValueError(
                        "Missing fallback prediction for a query below the threshold."
                    )
            correct += chosen_tool == prediction["expected_tool"]

        results.append(
            {
                "threshold": threshold,
                "accuracy": correct / total_queries,
                "laya_handled_count": laya_handled,
                "laya_handled_percentage": laya_handled / total_queries * 100,
                "query_count": total_queries,
            }
        )

    best_accuracy = max(row["accuracy"] for row in results)
    best_usage = max(
        row["laya_handled_percentage"]
        for row in results
        if row["accuracy"] == best_accuracy
    )
    selected = next(
        row
        for row in results
        if row["accuracy"] == best_accuracy
        and row["laya_handled_percentage"] == best_usage
    )
    return results, selected


def _write_results(results, selected):
    """Save the TRAIN sweep results and selected threshold."""
    json_path = OUTPUT_DIR / "threshold_sweep.json"
    csv_path = OUTPUT_DIR / "threshold_sweep.csv"
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(
            {
                "training_file": str(TRAIN_FILE.relative_to(PROJECT_ROOT)),
                "training_queries": results[0]["query_count"],
                "selection_rule": (
                    "Choose highest TRAIN accuracy; break ties by the highest "
                    "percentage handled by Laya."
                ),
                "selected_threshold": selected["threshold"],
                "selected_accuracy": selected["accuracy"],
                "selected_laya_handled_percentage": selected[
                    "laya_handled_percentage"
                ],
                "results": results,
            },
            output,
            indent=2,
        )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def _write_plot(results, selected):
    """Plot TRAIN accuracy against the percentage handled by Laya."""
    image_path = OUTPUT_DIR / "threshold_sweep.png"
    x_values = [row["laya_handled_percentage"] for row in results]
    y_values = [row["accuracy"] * 100 for row in results]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(x_values, y_values, marker="o")
    axis.scatter(
        [selected["laya_handled_percentage"]],
        [selected["accuracy"] * 100],
        color="red",
        label=f"Selected threshold: {selected['threshold']:.2f}",
        zorder=3,
    )
    axis.set_xlabel("TRAIN queries handled by Laya without an LLM call (%)")
    axis.set_ylabel("TRAIN routing accuracy (%)")
    axis.set_title("Hybrid confidence-threshold sweep (TRAIN split)")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    return image_path


def run_sweep():
    """Run the threshold sweep exclusively on TRAIN queries and save artifacts."""
    records = load_train_records()
    cached_predictions = collect_predictions(records)
    results, selected = sweep_thresholds(cached_predictions)
    json_path, csv_path = _write_results(results, selected)
    image_path = _write_plot(results, selected)
    print(
        f"Selected threshold {selected['threshold']:.2f}: "
        f"TRAIN accuracy {selected['accuracy']:.1%}, "
        f"Laya handled {selected['laya_handled_percentage']:.1f}% "
        f"of TRAIN queries without an LLM call."
    )
    print(f"Saved sweep results to {json_path} and {csv_path}")
    print(f"Saved trade-off plot to {image_path}")
    return results, selected


if __name__ == "__main__":
    run_sweep()
