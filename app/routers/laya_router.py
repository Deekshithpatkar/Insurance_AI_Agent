"""Laya decision-model router."""

import time

from laya import Router

laya = Router()

LAYA_TOOL_QUESTION = {
    "select_tool": {
        "type": "choice",
        "instructions": "Select the single most appropriate tool to answer the user's healthcare query.",
        "criteria": {
            "check_patient_policy": "Questions inquiring about patient diagnosis, whether a treatment is covered by insurance, or policy limits.",
            "calculate_out_of_pocket": "Requests asking how much the patient owes out of pocket, copays, or balance remaining after insurance.",
            "request_prior_authorization": "Requests to submit, file, or request prior authorization for specialized or uncovered medical procedures.",
            "escalate_to_human_specialist": "Patient disputes, emergency complaints, urgent issues, or requests outside the available insurance tools.",
        },
    }
}


def route(query: str) -> dict:
    """Use Laya to select a tool and return its confidence and latency."""
    start = time.perf_counter()
    prediction = laya.predict(query, LAYA_TOOL_QUESTION)
    latency_ms = (time.perf_counter() - start) * 1000
    decision = prediction.get("answers", {}).get("select_tool", {})
    usage = prediction.get("usage", {})

    return {
        "tool": decision.get("choice", "none"),
        "confidence": float(decision.get("confidence", 0.0)),
        "latency_ms": round(latency_ms, 2),
        "output_tokens": usage.get("output_tokens", 0),
    }
