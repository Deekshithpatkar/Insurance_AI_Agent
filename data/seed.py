"""Generate a reproducible sample database; running this replaces both seed tables."""

import os
import random

import psycopg2
from dotenv import load_dotenv
from faker import Faker

SEED = 2026
PATIENT_COUNT = 500

TREATMENTS = [
    ("MRI scan", 2400),
    ("CT scan", 1800),
    ("X-ray", 350),
    ("Ultrasound", 700),
    ("Echocardiogram", 1600),
    ("Colonoscopy", 3200),
    ("Endoscopy", 2800),
    ("Physical therapy", 2400),
    ("Occupational therapy", 2600),
    ("Inpatient rehabilitation", 32000),
    ("Outpatient surgery", 12000),
    ("Cardiac bypass surgery", 110000),
    ("Knee replacement", 48000),
    ("Hip replacement", 52000),
    ("Cataract surgery", 8500),
    ("Chemotherapy", 45000),
    ("Radiation therapy", 38000),
    ("Dialysis", 42000),
    ("Insulin therapy", 6000),
    ("Maternity care", 18000),
    ("Fertility treatment", 16000),
    ("Mental health counseling", 3000),
    ("Psychiatric hospitalization", 26000),
    ("Prescription medication", 2400),
    ("Emergency room visit", 7500),
    ("Ambulance transport", 2200),
    ("Home nursing", 14000),
    ("Skilled nursing facility", 36000),
    ("Sleep study", 2400),
    ("CPAP therapy", 1800),
    ("Genetic testing", 5200),
    ("Organ transplant", 240000),
    ("Stem cell therapy", 65000),
    ("Experimental gene therapy", 180000),
    ("Chiropractic care", 1800),
    ("Acupuncture", 1500),
    ("Dental surgery", 9000),
    ("Hearing aids", 5200),
    ("Vision correction surgery", 6000),
    ("Neurostimulation implant", 75000),
]

DIAGNOSES = [
    "Hypertension",
    "Type 2 diabetes",
    "Coronary artery disease",
    "Chronic back pain",
    "Osteoarthritis",
    "Asthma",
    "Sleep apnea",
    "Chronic kidney disease",
    "Cataracts",
    "Anxiety disorder",
    "Major depressive disorder",
    "Breast cancer",
    "Colorectal cancer",
    "Infertility",
    "Hearing loss",
    "Vision impairment",
    "Hip fracture",
    "Knee injury",
    "Heart valve disease",
    "Autoimmune disorder",
]


def build_seed_data():
    """Generate deterministic patient and policy rows, including required edge cases."""
    fake = Faker()
    fake.seed_instance(SEED)
    rng = random.Random(SEED)

    uncovered_treatments = {20, 32, 34, 35, 36}
    treatments_without_rules = {33, 39}
    policy_rows = []

    for index, (treatment, typical_cost) in enumerate(TREATMENTS):
        if index in treatments_without_rules:
            continue
        covered = index not in uncovered_treatments
        if not covered:
            limit = 0.0
        elif index == 9:
            limit = 8000.0
        else:
            limit = round(typical_cost * rng.uniform(0.4, 1.4), 2)
        policy_rows.append((treatment, covered, limit))

    patients = []
    used_names = set()
    for index in range(PATIENT_COUNT):
        if index < 8:
            name = f"Alex {fake.last_name()}"
        else:
            name = fake.name()
        while name.casefold() in used_names:
            name = fake.name()
        used_names.add(name.casefold())

        if index == 0:
            treatment_index = 9
        elif index == 1:
            treatment_index = 20
        elif index == 2:
            treatment_index = 33
        else:
            treatment_index = rng.randrange(len(TREATMENTS))

        treatment, typical_cost = TREATMENTS[treatment_index]
        if index == 0:
            cost = 44000.0
        else:
            cost = round(typical_cost * rng.uniform(0.7, 1.6), 2)

        patients.append(
            (
                name,
                rng.randint(18, 90),
                rng.choice(DIAGNOSES),
                treatment,
                cost,
            )
        )

    return patients, policy_rows


def seed_database(connection):
    """Create or update the seed schema and replace its rows deterministically."""
    patients, policy_rows = build_seed_data()
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255),
                age INTEGER,
                diagnosis VARCHAR(255),
                treatment VARCHAR(255),
                treatment_cost NUMERIC(12, 2)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_rules (
                id SERIAL PRIMARY KEY,
                treatment VARCHAR(255) NOT NULL UNIQUE,
                covered BOOLEAN NOT NULL,
                coverage_limit NUMERIC(12, 2)
            )
            """
        )
        cursor.execute(
            "ALTER TABLE patients ADD COLUMN IF NOT EXISTS treatment_cost NUMERIC(12, 2)"
        )
        cursor.execute("TRUNCATE TABLE patients, policy_rules RESTART IDENTITY")
        cursor.executemany(
            """
            INSERT INTO patients (name, age, diagnosis, treatment, treatment_cost)
            VALUES (%s, %s, %s, %s, %s)
            """,
            patients,
        )
        cursor.executemany(
            """
            INSERT INTO policy_rules (treatment, covered, coverage_limit)
            VALUES (%s, %s, %s)
            """,
            policy_rows,
        )
    finally:
        cursor.close()

    return {"patients": len(patients), "treatments": len(TREATMENTS), "policy_rules": len(policy_rows)}


def main():
    """Seed PostgreSQL and report the number of rows written."""
    load_dotenv()
    connection = psycopg2.connect(
        database=os.getenv("DB_NAME", "healthcare_agent"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )
    try:
        counts = seed_database(connection)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    print(
        f"Seeded {counts['patients']} patients, {counts['treatments']} treatments, "
        f"and {counts['policy_rules']} policy rules."
    )


if __name__ == "__main__":
    main()
