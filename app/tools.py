import hashlib

from langchain_core.tools import tool
from .db_connection import get_connection


@tool
def check_patient_policy(patient_name: str) -> dict:
    """Retrieve patient details, diagnosis, prescribed treatment, and insurance policy coverage rules from the database."""
    connection = get_connection()
    cursor = connection.cursor()

    query = """
        SELECT
            p.name,
            p.age,
            p.diagnosis,
            p.treatment,
            p.treatment_cost,
            pr.covered,
            pr.coverage_limit
        FROM patients p
        LEFT JOIN policy_rules pr
            ON LOWER(p.treatment) = LOWER(pr.treatment)
        WHERE LOWER(p.name) = LOWER(%s);
    """

    try:
        cursor.execute(query, (patient_name,))
        result = cursor.fetchone()
    finally:
        cursor.close()

    if result is None:
        return {"status": "patient_not_found", "message": f"No patient named '{patient_name}' found in database."}

    name, age, diagnosis, treatment, cost, covered, limit = result

    if covered is None:
        return {
            "status": "policy_rule_not_found",
            "patient": name,
            "diagnosis": diagnosis,
            "treatment": treatment,
            "treatment_cost": float(cost) if cost is not None else None,
            "message": f"No insurance policy rule found for treatment '{treatment}'."
        }

    return {
        "status": "success",
        "patient": name,
        "age": age,
        "diagnosis": diagnosis,
        "treatment": treatment,
        "treatment_cost": float(cost) if cost is not None else None,
        "covered": covered,
        "coverage_limit": float(limit) if limit is not None else None
    }


def _deterministic_id(prefix: str, value: str) -> str:
    """Build a stable identifier from a prefix and its input value."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


@tool
def calculate_out_of_pocket(treatment_cost: float, coverage_limit: float) -> dict:
    """Calculate the remaining out-of-pocket amount a patient must pay after insurance coverage limit is deducted."""
    try:
        cost = float(treatment_cost)
        limit = float(coverage_limit)
        remaining = max(0.0, cost - limit)
        is_fully_covered = remaining == 0.0

        return {
            "status": "success",
            "treatment_cost": cost,
            "coverage_limit": limit,
            "patient_owes": remaining,
            "fully_covered": is_fully_covered,
            "message": f"Patient pays ${remaining:.2f} out of pocket." if remaining > 0 else "Treatment is 100% covered by the policy limit."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def request_prior_authorization(patient_name: str, treatment: str) -> dict:
    """Submit an expedited prior authorization review request for expensive, specialized, or uncovered treatments."""
    return {
        "status": "submitted",
        "patient": patient_name,
        "treatment": treatment,
        "auth_ticket_id": _deterministic_id("AUTH", patient_name + treatment),
        "message": f"Prior authorization request successfully submitted for {patient_name}'s {treatment}. Decision pending clinical review."
    }


@tool
def escalate_to_human_specialist(patient_name: str, issue_description: str) -> dict:
    """Escalate urgent medical emergencies, disputed claims, or patient grievances directly to a human insurance specialist."""
    return {
        "status": "escalated",
        "patient": patient_name,
        "issue": issue_description,
        "escalation_id": _deterministic_id("ESC", patient_name + issue_description),
        "message": f"Claim issue for {patient_name} has been escalated to a human specialist for priority resolution."
    }


# Tool lookup map and tool list
TOOL_MAP = {
    "check_patient_policy": check_patient_policy,
    "calculate_out_of_pocket": calculate_out_of_pocket,
    "request_prior_authorization": request_prior_authorization,
    "escalate_to_human_specialist": escalate_to_human_specialist
}

tools = [
    check_patient_policy,
    calculate_out_of_pocket,
    request_prior_authorization,
    escalate_to_human_specialist
]