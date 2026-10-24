"""Confidence-gated Laya router with Qwen fallback."""

import time

from . import laya_router, llm_router

CONFIDENCE_THRESHOLD = 0.75
VALID_TOOLS = {
    "check_patient_policy",
    "calculate_out_of_pocket",
    "request_prior_authorization",
    "escalate_to_human_specialist",
}


def route(query: str) -> dict:
    """Use Laya above the confidence threshold; otherwise fall back to Qwen."""
    start = time.perf_counter()
    laya_result = laya_router.route(query)

    if (
        laya_result["tool"] in VALID_TOOLS
        and laya_result["confidence"] >= CONFIDENCE_THRESHOLD
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
