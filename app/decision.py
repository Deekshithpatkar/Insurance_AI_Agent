import time
from typing import Dict, Any

from laya import Router
from .tools import check_patient_policy

# Initialize Laya Decision Model Router (System 1)
router = Router()

# Define typed decision questions for healthcare claims
CLAIM_DECISION_QUESTIONS: Dict[str, Any] = {
    "claim_decision": {
        "type": "choice",
        "instructions": "Determine whether the medical treatment claim should be approved, denied, or requires prior authorization.",
        "criteria": {
            "approved": "The treatment is covered by policy rules and within the coverage budget limit.",
            "denied": "The treatment is explicitly not covered by the insurance policy or patient record is missing.",
            "requires_prior_authorization": "The cost exceeds the coverage limit or requires doctor justification."
        }
    },
    "clinical_urgency": {
        "type": "choice",
        "instructions": "Assess the clinical urgency and priority level of the claim.",
        "criteria": {
            "routine": "Standard planned treatment or scheduled procedure.",
            "urgent": "Time-sensitive medical condition requiring prompt intervention.",
            "critical": "Severe, life-threatening emergency requiring immediate authorization."
        }
    },
    "is_compliant": {
        "type": "noul",
        "instructions": "Does this claim fully comply with policy coverage terms?"
    }
}


def extract_patient_name(user_query: str) -> str:
    """Extract patient name from a natural language query or return clean input."""
    # First check if the raw query matches a patient in the database
    from .db_connection import get_connection
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM patients")
        all_patients = [row[0] for row in cur.fetchall()]
        cur.close()

        # Check if any known patient name is mentioned in the query
        query_lower = user_query.lower()
        for p in all_patients:
            if p.lower() in query_lower:
                return p
    except Exception:
        pass

    # Fallback: clean words
    cleaned = user_query.strip(" ?.!/\\")
    return cleaned


def evaluate_claim_with_laya(user_input: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Pure Decision Model (Laya) execution:
    1. Extracts patient entity from receptionist query
    2. Fetches factual patient & policy data from PostgreSQL
    3. Runs a single non-autoregressive neural forward pass
    4. Outputs exact choices, probabilities, and confidence scores (0 generated tokens)
    """
    patient_name = extract_patient_name(user_input)

    if verbose:
        print(f"\n[Query Received] '{user_input}'")
        print(f"[Entity Extracted] Identified Patient: '{patient_name}'")
        print(f"[Decision Model] Fetching database record for: '{patient_name}'...")

    # Step 1: Query PostgreSQL
    db_result = check_patient_policy.invoke({"patient_name": patient_name})

    if verbose:
        print(f"[Decision Model] DB Result: {db_result}")

    if db_result.get("status") == "patient_not_found":
        state_text = f"Patient '{patient_name}' was not found in the healthcare database records."
    else:
        patient = db_result.get("patient", patient_name)
        age = db_result.get("age", "Unknown")
        diag = db_result.get("diagnosis", "Unknown")
        treat = db_result.get("treatment", "Unknown")
        covered = db_result.get("covered", False)
        limit = db_result.get("coverage_limit", 0)

        state_text = (
            f"Patient: {patient}, Age: {age}. "
            f"Diagnosis: {diag}. "
            f"Prescribed Treatment: {treat}. "
            f"Insurance Covered: {covered}. "
            f"Policy Coverage Limit: ${limit}."
        )

    # Step 2: Run Laya Neural Decision (System 1)
    if verbose:
        print(f"[Decision Model] Running Laya neural forward pass (non-autoregressive)...")

    start_time = time.perf_counter()
    raw_prediction = router.predict(state_text, CLAIM_DECISION_QUESTIONS)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    answers = raw_prediction.get("answers", {})
    usage = raw_prediction.get("usage", {})

    claim_dec = answers.get("claim_decision", {})
    urgency = answers.get("clinical_urgency", {})
    compliant = answers.get("is_compliant", {})

    result = {
        "patient_name": patient_name,
        "state_context": state_text,
        "decision": claim_dec.get("choice", "unknown"),
        "decision_confidence": claim_dec.get("confidence", 0.0),
        "decision_probabilities": claim_dec.get("probabilities", {}),
        "urgency": urgency.get("choice", "unknown"),
        "urgency_confidence": urgency.get("confidence", 0.0),
        "compliance_probability": compliant.get("noul", 0.0),
        "latency_ms": round(elapsed_ms, 2),
        "output_tokens": usage.get("output_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0)
    }

    if verbose:
        print("\n" + "=" * 60)
        print("          LAYA DECISION MODEL RESULTS (SYSTEM 1)")
        print("=" * 60)
        print(f"Decision          : {result['decision'].upper()} (Confidence: {result['decision_confidence']:.2%})")
        print(f"Probabilities     : {result['decision_probabilities']}")
        print(f"Urgency Level     : {result['urgency'].upper()} (Confidence: {result['urgency_confidence']:.2%})")
        print(f"Compliance Prob   : {result['compliance_probability']:.2%}")
        print(f"Execution Latency : {result['latency_ms']} ms")
        print(f"Tokens Generated  : {result['output_tokens']} (Zero tokens - pure neural classification!)")
        print("=" * 60)

    return result


def main():
    print("==========================================================")
    print("   Healthcare Claims Decision Model (Powered by Laya)")
    print("   Non-Autoregressive System 1 Neural Classification")
    print("==========================================================")
    print("Ask any claim question or enter a patient name (type 'exit' to quit).\n")
    print("Example: 'Is Rahul\\'s treatment covered in the healthcare policy?'\n")

    while True:
        try:
            query = input("Receptionist : ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting decision model. Goodbye!")
                break

            evaluate_claim_with_laya(query)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
