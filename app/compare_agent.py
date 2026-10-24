import os
import json
import time
from typing import Dict, Any, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from laya import Router

from .tools import tools, TOOL_MAP

# Benchmark data file to store background telemetry
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_FILE = os.path.join(project_root, "benchmark_data.json")

# Initialize models
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
llm_with_tools = llm.bind_tools(tools)
laya_router = Router()

# Laya tool selection question definition
LAYA_TOOL_QUESTION = {
    "select_tool": {
        "type": "choice",
        "instructions": "Select the single most appropriate tool to answer the user's healthcare query.",
        "criteria": {
            "check_patient_policy": "Questions inquiring about patient diagnosis, whether a treatment is covered by insurance, or policy limits.",
            "calculate_out_of_pocket": "Requests asking how much the patient owes out of pocket, copays, or balance remaining after insurance.",
            "request_prior_authorization": "Requests to submit, file, or request prior authorization for specialized or uncovered medical procedures.",
            "escalate_to_human_specialist": "Patient disputes, emergency complaints, or urgent issues requiring human specialist intervention."
        }
    }
}


def _output_tokens(message) -> int:
    """Read generated-token usage from LangChain metadata or Ollama metadata."""
    usage = getattr(message, "usage_metadata", None) or {}
    if "output_tokens" in usage:
        return usage["output_tokens"]
    response_metadata = getattr(message, "response_metadata", None) or {}
    return response_metadata.get("eval_count", 0)


def run_genai_agent(user_query: str) -> Tuple[str, Dict[str, Any]]:
    """
    GenAI Agent (Qwen 2.5):
    - Uses autoregressive tool-calling to understand the question and invoke the tool.
    - Generates a plain-text natural language response for the user.
    - Measures latency and token consumption for background storage.
    """
    system_prompt = (
        "You are a helpful healthcare insurance assistant. "
        "Answer the user's question directly and concisely in plain text. "
        "Do not include internal tool details, JSON blobs, or technical jargon in your final response."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    selected_tool_name = "none"
    tool_output = None
    tokens_generated = 0
    start_time = time.perf_counter()

    # Step 1: Model decides tool to invoke
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tokens_generated += _output_tokens(response)

    # Step 2: Execute selected tool if requested
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        selected_tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Execute tool
        tool_fn = TOOL_MAP.get(selected_tool_name)
        if tool_fn:
            result = tool_fn.invoke(tool_args)
        else:
            result = f"Tool {selected_tool_name} not found."
        tool_output = result

        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

        # Step 3: Model summarizes final answer in plain text
        final_response = llm.invoke(messages)
        tokens_generated += _output_tokens(final_response)

        plain_text_answer = final_response.content.strip()
    else:
        plain_text_answer = response.content.strip()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    telemetry = {
        "model": "Qwen 2.5 (GenAI)",
        "tool_selected": selected_tool_name,
        "latency_ms": round(elapsed_ms, 2),
        "tokens_generated": tokens_generated,
        "answer": plain_text_answer,
        "tool_output": tool_output
    }

    return plain_text_answer, telemetry


def run_laya_decision(user_query: str) -> Dict[str, Any]:
    """
    Decision Model (Laya):
    - Uses non-autoregressive neural forward pass to pick the best tool.
    - Zero tokens generated; pure calibrated classification.
    """
    start_time = time.perf_counter()
    prediction = laya_router.predict(user_query, LAYA_TOOL_QUESTION)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    answers = prediction.get("answers", {})
    tool_decision = answers.get("select_tool", {})
    usage = prediction.get("usage", {})

    chosen_tool = tool_decision.get("choice", "unknown")
    confidence = tool_decision.get("confidence", 0.0)
    probabilities = tool_decision.get("probabilities", {})

    return {
        "model": "Laya (Decision Model)",
        "tool_selected": chosen_tool,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "latency_ms": round(elapsed_ms, 2),
        "tokens_generated": usage.get("output_tokens", 0)
    }


def save_background_telemetry(query: str, genai_data: Dict[str, Any], laya_data: Dict[str, Any]) -> None:
    """Save background metrics to benchmark_data.json for plotting and comparison."""
    data = []
    if os.path.exists(BENCHMARK_FILE):
        try:
            with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []

    record = {
        "id": len(data) + 1,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "genai": genai_data,
        "laya": laya_data,
        "agreement": genai_data["tool_selected"] == laya_data["tool_selected"]
    }

    data.append(record)

    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def process_user_query(query: str) -> str:
    """
    Main processing entrypoint:
    1. Runs GenAI model and obtains user plain-text response.
    2. Runs Laya decision model in the background for tool selection.
    3. Saves comparison telemetry silently to benchmark_data.json.
    4. Returns ONLY the clean plain text for the user.
    """
    # 1. GenAI model creates the conversational answer and picks its tool
    plain_text_response, genai_telemetry = run_genai_agent(query)

    # 2. Laya decision model selects tool in the background
    laya_telemetry = run_laya_decision(query)

    # 3. Save background comparison
    save_background_telemetry(query, genai_telemetry, laya_telemetry)

    # 4. Return ONLY plain text to user
    return plain_text_response


def main():
    print("=== Healthcare Assistant Ready ===")
    print("Type your question below (type 'exit' to quit).\n")

    while True:
        try:
            query = input("You : ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            response = process_user_query(query)
            print(f"\n{response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
