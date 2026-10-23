import re
import time
from typing import Dict, Any, Tuple

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from laya import Router

from .tools import TOOL_MAP
from .db_connection import get_connection

# Initialize Models
laya_router = Router()
llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

# 1. Laya Tool Routing Question Definition (System 1)
LAYA_ROUTING_QUESTION = {
    "select_tool": {
        "type": "choice",
        "instructions": "Select the single best tool to resolve the user's healthcare request.",
        "criteria": {
            "check_patient_policy": "Questions inquiring about patient diagnosis, whether a treatment is covered by insurance, or policy limits.",
            "calculate_out_of_pocket": "Requests to calculate out-of-pocket costs, remaining patient balances, or copays given treatment costs and limits.",
            "request_prior_authorization": "Requests to file or submit prior authorization for high-cost, specialized, or uncovered procedures.",
            "escalate_to_human_specialist": "Patient disputes, emergency complaints, or urgent grievances requiring human specialist resolution."
        }
    }
}

# 2. Laya Claim Verification Question Definition
LAYA_VERIFICATION_QUESTION = {
    "claim_decision": {
        "type": "choice",
        "instructions": "Determine if the claim is approved, denied, or requires prior authorization.",
        "criteria": {
            "approved": "The treatment is covered by policy rules and within the coverage limit.",
            "denied": "The treatment is explicitly not covered by policy rules or patient is missing.",
            "requires_prior_authorization": "Treatment exceeds coverage limit or requires specialized clinical review."
        }
    },
    "is_compliant": {
        "type": "noul",
        "instructions": "Does this claim comply with policy coverage terms?"
    }
}


def extract_patient_name(query: str) -> str:
    """Find known patient name from PostgreSQL or return default."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM patients")
        names = [r[0] for r in cur.fetchall()]
        cur.close()

        query_lower = query.lower()
        for name in names:
            if name.lower() in query_lower:
                return name
    except Exception:
        pass

    # Basic fallback pattern
    words = re.findall(r'[A-Za-z]+', query)
    return words[0] if words else "Unknown"


def extract_numbers(query: str) -> Tuple[float, float]:
    """Extract two dollar or numeric amounts from query (e.g. cost and limit)."""
    nums = [float(n.replace(',', '')) for n in re.findall(r'\$?\b\d+(?:,\d+)*(?:\.\d+)?\b', query)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    elif len(nums) == 1:
        return nums[0], 0.0
    return 0.0, 0.0


def run_hybrid_agent(user_query: str, verbose: bool = True) -> str:
    """
    The Collaborative Hybrid Pipeline:
    1. Laya (System 1) -> Fast neural routing to select the tool (0 tokens).
    2. Tool Execution  -> PostgreSQL / Business logic fetches live data.
    3. Laya (System 1) -> Neural policy verification & compliance check (0 tokens).
    4. Qwen (System 2) -> Generates the final plain-text response strictly from verified facts.
    """
    total_start = time.perf_counter()

    if verbose:
        print("\n" + "=" * 65)
        print("                HYBRID AGENT EXECUTION TRACE")
        print("=" * 65)
        print(f"[Receptionist Query] '{user_query}'")

    # --- Step 1: Laya Neural Tool Routing ---
    laya_start = time.perf_counter()
    routing_pred = laya_router.predict(user_query, LAYA_ROUTING_QUESTION)
    laya_routing_ms = (time.perf_counter() - laya_start) * 1000

    tool_ans = routing_pred.get("answers", {}).get("select_tool", {})
    chosen_tool = tool_ans.get("choice", "check_patient_policy")
    tool_conf = tool_ans.get("confidence", 0.0)

    if verbose:
        print(f"[Step 1: Laya Router] Chose Tool: '{chosen_tool}' (Confidence: {tool_conf:.2%}, {laya_routing_ms:.1f}ms, 0 tokens)")

    # --- Step 2: Execute Selected Tool ---
    tool_fn = TOOL_MAP.get(chosen_tool, TOOL_MAP["check_patient_policy"])
    tool_data = {}

    if chosen_tool == "check_patient_policy":
        patient_name = extract_patient_name(user_query)
        tool_data = tool_fn.invoke({"patient_name": patient_name})

    elif chosen_tool == "calculate_out_of_pocket":
        cost, limit = extract_numbers(user_query)
        # If limit wasn't in query, look up patient in DB
        if limit == 0.0:
            patient_name = extract_patient_name(user_query)
            p_info = TOOL_MAP["check_patient_policy"].invoke({"patient_name": patient_name})
            limit = p_info.get("coverage_limit", 0.0) or 0.0

        tool_data = tool_fn.invoke({"treatment_cost": cost, "coverage_limit": limit})

    elif chosen_tool == "request_prior_authorization":
        patient_name = extract_patient_name(user_query)
        p_info = TOOL_MAP["check_patient_policy"].invoke({"patient_name": patient_name})
        treatment = p_info.get("treatment", "Specialized Care") if p_info.get("status") == "success" else "Requested Treatment"
        tool_data = tool_fn.invoke({"patient_name": patient_name, "treatment": treatment})

    elif chosen_tool == "escalate_to_human_specialist":
        patient_name = extract_patient_name(user_query)
        tool_data = tool_fn.invoke({"patient_name": patient_name, "issue_description": user_query})

    if verbose:
        print(f"[Step 2: Tool Execution] Result: {tool_data}")

    # --- Step 3: Laya Neural Verification Gate ---
    decision_summary = "verified"
    if chosen_tool == "check_patient_policy" and tool_data.get("status") == "success":
        state_text = (
            f"Patient: {tool_data.get('patient')}. "
            f"Diagnosis: {tool_data.get('diagnosis')}. "
            f"Treatment: {tool_data.get('treatment')}. "
            f"Insurance Covered: {tool_data.get('covered')}. "
            f"Coverage Limit: ${tool_data.get('coverage_limit')}."
        )
        laya_verify_pred = laya_router.predict(state_text, LAYA_VERIFICATION_QUESTION)
        v_ans = laya_verify_pred.get("answers", {})
        dec = v_ans.get("claim_decision", {}).get("choice", "approved")
        compliance = v_ans.get("is_compliant", {}).get("noul", 1.0)
        decision_summary = f"{dec.upper()} (Compliance: {compliance:.1%})"

        if verbose:
            print(f"[Step 3: Laya Verification] Claim Decision: {decision_summary} (0 tokens)")

    # --- Step 4: Qwen (LLM) Generates Final Plain Text Response ---
    qwen_start = time.perf_counter()
    prompt = f"""
    You are a professional healthcare assistant.
    Answer the user's question directly in 1-2 concise, friendly sentences in plain text.

    User Question: {user_query}
    Verified Tool Facts: {tool_data}
    Neural Decision Status: {decision_summary}

    Rules:
    - Respond ONLY in plain text.
    - Rely strictly on the verified facts above.
    - Do not invent amounts, diagnosis, or tool names.
    """

    llm_response = llm.invoke([
        SystemMessage(content="You are a healthcare assistant. Output only plain text."),
        HumanMessage(content=prompt)
    ])
    qwen_ms = (time.perf_counter() - qwen_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    plain_text_answer = llm_response.content.strip()

    if verbose:
        print(f"[Step 4: Qwen LLM Communication] Completed in {qwen_ms:.1f}ms")
        print(f"[Total Pipeline Latency] {total_ms:.1f}ms")
        print("=" * 65 + "\n")

    return plain_text_answer


def main():
    print("==========================================================")
    print("   Hybrid Healthcare AI Agent (Laya + Qwen 2.5)")
    print("   System 1 (Laya Decisions) + System 2 (Qwen Language)")
    print("==========================================================")
    print("Ask any question in plain English (type 'exit' to quit).\n")

    while True:
        try:
            query = input("You : ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            answer = run_hybrid_agent(query, verbose=True)
            print(f"Agent:\n{answer}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
