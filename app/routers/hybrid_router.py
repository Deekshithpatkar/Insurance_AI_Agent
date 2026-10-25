"""Confidence-gated Laya router with Qwen fallback."""

import time

from . import laya_router, llm_router

CONFIDENCE_THRESHOLD = 0.30
VALID_TOOLS = {
    "check_patient_policy",
    "calculate_out_of_pocket",
    "request_prior_authorization",
    "escalate_to_human_specialist",
}


def route_with_threshold(query: str, threshold: float) -> dict:
    """Route with a supplied confidence threshold and return combined metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The confidence threshold must be between 0 and 1.")

    start = time.perf_counter()
    laya_result = laya_router.route(query)

    if (
        laya_result["tool"] in VALID_TOOLS
        and laya_result["confidence"] >= threshold
    ):
        return {
            "tool": laya_result["tool"],
            "confidence": laya_result["confidence"],
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "output_tokens": laya_result["output_tokens"],
        }

    llm_result = llm_router.route(query)
    return {
        "tool": llm_result["tool"],
        "confidence": llm_result["confidence"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "output_tokens": llm_result["output_tokens"],
    }


def route(query: str) -> dict:
    """Use the configured Laya confidence threshold; otherwise fall back to Qwen."""
    return route_with_threshold(query, CONFIDENCE_THRESHOLD)
