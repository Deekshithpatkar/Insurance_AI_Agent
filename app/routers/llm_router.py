"""Qwen tool-calling router."""

import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from ..tools import tools

llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def _output_tokens(message) -> int:
    """Read output-token usage from LangChain or Ollama metadata."""
    usage = getattr(message, "usage_metadata", None) or {}
    if "output_tokens" in usage:
        return usage["output_tokens"]
    response_metadata = getattr(message, "response_metadata", None) or {}
    return response_metadata.get("eval_count", 0)


def route(query: str) -> dict:
    """Ask Qwen to select a tool and return its routing metrics."""
    messages = [
        SystemMessage(
            content=(
                "Choose exactly one available tool for the user's request. "
                "For requests outside the insurance tools, choose "
                "escalate_to_human_specialist. Do not answer the request."
            )
        ),
        HumanMessage(content=query),
    ]

    start = time.perf_counter()
    response = llm_with_tools.invoke(messages)
    latency_ms = (time.perf_counter() - start) * 1000
    tool = response.tool_calls[0]["name"] if response.tool_calls else "none"

    return {
        "tool": tool,
        "confidence": 0.0,
        "latency_ms": round(latency_ms, 2),
        "output_tokens": _output_tokens(response),
    }
