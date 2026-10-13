from langchain_core.tools import tool
try:
    from app.db_connection import connection
except ImportError:
    from db_connection import connection


@tool
def check_patient_policy(patient_name: str):
    """Retrieve a patient's details and check their treatment against the policy rules in PostgreSQL."""

    cursor = connection.cursor()

    query = """
        SELECT
            p.name,
            p.age,
            p.diagnosis,
            p.treatment,
            pr.covered,
            pr.coverage_limit
        FROM patients p
        LEFT JOIN policy_rules pr
            ON LOWER(p.treatment) = LOWER(pr.treatment)
        WHERE LOWER(p.name) = LOWER(%s);
    """

    cursor.execute(query, (patient_name,))
    result = cursor.fetchone()
    cursor.close()

    if result is None:
        return {"status": "patient_not_found"}

    name, age, diagnosis, treatment, covered, limit = result

    if covered is None:
        return {
            "status": "policy_rule_not_found",
            "patient": name,
            "diagnosis": diagnosis,
            "treatment": treatment
        }

    return {
        "status": "success",
        "patient": name,
        "age": age,
        "diagnosis": diagnosis,
        "treatment": treatment,
        "covered": covered,
        "coverage_limit": float(limit) if limit is not None else None
    }


# Export list of tools for the agent
tools = [check_patient_policy]