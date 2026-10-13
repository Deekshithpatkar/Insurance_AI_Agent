import sys
import os
from langchain_core.tools import tool

# Ensure imports work whether executed directly or as a package
try:
    from app.db_connection import get_connection
except ImportError:
    from db_connection import get_connection

@tool
def patient_info(patient_name: str):
    """get patient details"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = "Select name, age, diagnosis, treatment from patients where lower(name) = lower(%s)"
        cursor.execute(query, (patient_name,))
        patient = cursor.fetchone()
        cursor.close()

        if patient is None:
            return f"No patient named {patient_name} found."

        return {
            'name': patient[0],
            'age': patient[1],
            'diagnosis': patient[2],
            'treatment': patient[3]
        }
    except Exception as e:
        return f"Database error: {str(e)}"

@tool
def policy_coverage(treatment: str):
    """check whether the treatment is covered by the policy"""
    covered_treatment = ["appendectomy", "chemotherapy", "radiation therapy"]

    if treatment.lower() in covered_treatment:
        return f"{treatment} covered by the policy"

    return f"{treatment} not covered by the policy"

@tool
def coverage_limit(treatment: str):
    """Get the maximum amount covered for a treatment."""
    limit = {
        "appendectomy": 5000,
        "chemotherapy": 30000,
        "radiation therapy": 10000
    }
    if treatment.lower() in limit:
        return limit[treatment.lower()]
    return 0

@tool
def calculate_remaining(treatment: str, treatment_cost: float, coverage_limit: float):
    """calculate how much patient has to pay"""
    remaining = treatment_cost - coverage_limit
    if remaining > 0:
        return f"patient has to pay {remaining}"
    return "the treatment is covered by the policy"

# List of tools provided to the agent
tools = [patient_info, policy_coverage, coverage_limit, calculate_remaining]
