"""Measure whether final answers preserve facts returned by their selected tool."""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from app.compare_agent import run_genai_agent
from app.db_connection import get_connection
from app.routers import hybrid_router, llm_router
from app.tools import TOOL_MAP, check_patient_policy
from data.seed import TREATMENTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = PROJECT_ROOT / "data" / "eval_test.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent
AMOUNT_KEYS = ("treatment_cost", "coverage_limit", "patient_owes")
KNOWN_TREATMENTS = [treatment for treatment, _ in TREATMENTS]
MONEY_PATTERN = re.compile(
    r"(?:\$\s*|USD\s*)(\d[\d,]*(?:\.\d{1,2})?)"
    r"|\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:USD|dollars?)\b",
    re.IGNORECASE,
)
POSITIVE_COVERAGE_PATTERN = re.compile(r"\b(?:covered|covers|coverage applies)\b", re.IGNORECASE)
NEGATIVE_COVERAGE_PATTERN = re.compile(
    r"\b(?:not\s+(?:be\s+)?covered|uncovered|does\s+not\s+cover|doesn't\s+cover|"
    r"is\s+not\s+covered|isn't\s+covered|denied)\b",
    re.IGNORECASE,
)


def load_test_records():
    """Read labeled queries from the held-out test split only."""
    with TEST_FILE.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _patient_for_query(query):
    """Find the exact seeded patient name mentioned in a query."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT name FROM patients WHERE name IS NOT NULL ORDER BY LENGTH(name) DESC")
        names = [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()

    query_lower = query.casefold()
    for name in names:
        if name.casefold() in query_lower:
            return name
    raise ValueError(
        "No patient from the database appears in this test query. "
        "Run `python data/seed.py` before the faithfulness evaluation."
    )


def _validate_seed_database(records):
    """Fail early when the Phase 1 schema or test patients are missing."""
    connection = get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'patients'
              AND column_name = 'treatment_cost'
            """
        )
        if cursor.fetchone() is None:
            raise RuntimeError(
                "The database is missing the treatment_cost column. "
                "Run `python data/seed.py` before the faithfulness evaluation."
            )

        cursor.execute("SELECT name FROM patients WHERE name IS NOT NULL")
        names = [row[0] for row in cursor.fetchall()]
    finally:
        cursor.close()

    if not names:
        raise RuntimeError(
            "The patients table is empty. Run `python data/seed.py` before "
            "the faithfulness evaluation."
        )

    missing = [
        record["id"]
        for record in records
        if not any(name.casefold() in record["query"].casefold() for name in names)
    ]
    if missing:
        raise RuntimeError(
            f"The database is missing patients used by test query ids {missing[:5]}. "
            "Run `python data/seed.py` before the faithfulness evaluation."
        )


def _execute_hybrid_tool(tool_name, query):
    """Execute the hybrid-selected tool using patient facts from the seeded database."""
    patient_name = _patient_for_query(query)
    patient_facts = check_patient_policy.invoke({"patient_name": patient_name})

    if tool_name == "check_patient_policy":
        return patient_facts
    if tool_name == "calculate_out_of_pocket":
        cost = patient_facts.get("treatment_cost")
        limit = patient_facts.get("coverage_limit")
        if patient_facts.get("status") != "success":
            return {
                "status": "calculation_unavailable",
                "treatment": patient_facts.get("treatment"),
                "message": patient_facts.get(
                    "message", "No policy rule is available for this treatment."
                ),
            }
        if cost is None or limit is None:
            return {
                "status": "calculation_unavailable",
                "treatment": patient_facts.get("treatment"),
                "message": f"Missing seeded cost or policy limit for {patient_name}.",
            }
        return TOOL_MAP[tool_name].invoke(
            {"treatment_cost": cost, "coverage_limit": limit}
        )
    if tool_name == "request_prior_authorization":
        return TOOL_MAP[tool_name].invoke(
            {
                "patient_name": patient_name,
                "treatment": patient_facts["treatment"],
            }
        )
    if tool_name == "escalate_to_human_specialist":
        return TOOL_MAP[tool_name].invoke(
            {"patient_name": patient_name, "issue_description": query}
        )
    raise ValueError(f"Router selected an unsupported tool: {tool_name}")


def _hybrid_answer(query):
    """Route with the hybrid system, execute its tool, and ask Qwen to summarize facts."""
    routing = hybrid_router.route(query)
    tool_output = _execute_hybrid_tool(routing["tool"], query)
    response = llm_router.llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the user briefly and accurately using only the supplied "
                    "tool result. Do not add unsupported facts."
                )
            ),
            HumanMessage(
                content=f"User query: {query}\nTool result: {json.dumps(tool_output)}"
            ),
        ]
    )
    return routing["tool"], tool_output, response.content.strip()


def _money_value(value):
    """Convert an amount to integer cents for tolerant comparison."""
    return round(float(value) * 100)


def check_faithfulness(tool_output, answer):
    """Check answer claims against available amounts, treatment, and coverage facts."""
    if not isinstance(tool_output, dict):
        return {
            "mismatch": None,
            "checked_fields": [],
            "unverified_fields": ["tool_output"],
            "mismatch_fields": [],
        }

    checked = []
    unverified = []
    mismatches = []
    answer_lower = answer.casefold()

    expected_amounts = {
        _money_value(tool_output[key])
        for key in AMOUNT_KEYS
        if tool_output.get(key) is not None
    }
    stated_amounts = {
        _money_value(next(value for value in match if value).replace(",", ""))
        for match in MONEY_PATTERN.findall(answer)
    }
    if stated_amounts:
        checked.append("amounts")
        if not stated_amounts.issubset(expected_amounts):
            mismatches.append("amounts")
    elif expected_amounts:
        unverified.append("amounts")

    treatment = tool_output.get("treatment")
    if treatment:
        mentioned_treatments = [
            known_treatment
            for known_treatment in KNOWN_TREATMENTS
            if known_treatment.casefold() in answer_lower
        ]
        if any(
            known_treatment.casefold() == treatment.casefold()
            for known_treatment in mentioned_treatments
        ):
            checked.append("treatment")
            if any(
                known_treatment.casefold() != treatment.casefold()
                for known_treatment in mentioned_treatments
            ):
                mismatches.append("treatment")
        elif mentioned_treatments:
            checked.append("treatment")
            mismatches.append("treatment")
        else:
            unverified.append("treatment")

    coverage_key = None
    if "covered" in tool_output:
        coverage_key = "covered"
    elif "fully_covered" in tool_output:
        coverage_key = "fully_covered"
    if coverage_key is not None:
        positive = bool(POSITIVE_COVERAGE_PATTERN.search(answer))
        negative = bool(NEGATIVE_COVERAGE_PATTERN.search(answer))
        if positive or negative:
            checked.append("coverage_status")
            if bool(tool_output[coverage_key]) != (positive and not negative):
                mismatches.append("coverage_status")
        else:
            unverified.append("coverage_status")

    return {
        "mismatch": bool(mismatches) if checked else None,
        "checked_fields": checked,
        "unverified_fields": unverified,
        "mismatch_fields": mismatches,
    }


def _collect_results(records):
    """Collect Qwen tool answers and hybrid tool answers for every test query."""
    results = []
    for record in records:
        answer, telemetry = run_genai_agent(record["query"])
        qwen_check = check_faithfulness(telemetry.get("tool_output"), answer)
        results.append(
            {
                "system": "qwen",
                "id": record["id"],
                "category": record["category"],
                "query": record["query"],
                "tool": telemetry["tool_selected"],
                "tool_output": telemetry.get("tool_output"),
                "answer": answer,
                **qwen_check,
            }
        )

        tool_name, tool_output, hybrid_answer = _hybrid_answer(record["query"])
        hybrid_check = check_faithfulness(tool_output, hybrid_answer)
        results.append(
            {
                "system": "hybrid",
                "id": record["id"],
                "category": record["category"],
                "query": record["query"],
                "tool": tool_name,
                "tool_output": tool_output,
                "answer": hybrid_answer,
                **hybrid_check,
            }
        )
    return results


def _summarize(results):
    """Calculate mismatch rates and disclose answers without verifiable facts."""
    grouped = defaultdict(list)
    for result in results:
        grouped[result["system"]].append(result)

    summary = {}
    for system, rows in grouped.items():
        evaluated = [row for row in rows if row["mismatch"] is not None]
        mismatches = sum(row["mismatch"] for row in evaluated)
        summary[system] = {
            "answers": len(rows),
            "answers_with_checkable_facts": len(evaluated),
            "unverifiable_answers": len(rows) - len(evaluated),
            "mismatches": mismatches,
            "mismatch_rate": (
                mismatches / len(evaluated) if evaluated else None
            ),
        }
    return summary


def _write_outputs(summary, results):
    """Save faithfulness aggregates and per-answer evidence."""
    json_path = OUTPUT_DIR / "faithfulness_results.json"
    csv_path = OUTPUT_DIR / "faithfulness_results.csv"
    with json_path.open("w", encoding="utf-8") as output:
        json.dump({"summary": summary, "results": results}, output, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        columns = [
            "system",
            "id",
            "category",
            "query",
            "tool",
            "tool_output",
            "answer",
            "mismatch",
            "checked_fields",
            "unverified_fields",
            "mismatch_fields",
        ]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = result.copy()
            row["tool_output"] = json.dumps(row["tool_output"], ensure_ascii=False)
            for key in ("checked_fields", "unverified_fields", "mismatch_fields"):
                row[key] = json.dumps(row[key])
            writer.writerow(row)
    return json_path, csv_path


def run_faithfulness():
    """Run tool-answer faithfulness checks on the test split and save the results."""
    records = load_test_records()
    _validate_seed_database(records)
    results = _collect_results(records)
    summary = _summarize(results)
    json_path, csv_path = _write_outputs(summary, results)
    print(json.dumps(summary, indent=2))
    print(f"Saved faithfulness results to {json_path}")
    print(f"Saved per-answer evidence to {csv_path}")
    return summary


if __name__ == "__main__":
    run_faithfulness()
