"""Build a synthetic, human-review-required tool-routing dataset and fixed splits."""

import json
import random

from data.seed import build_seed_data

OUTPUT_PATH = "data"
SPLIT_SEED = 2026

TOOLS = (
    "check_patient_policy",
    "calculate_out_of_pocket",
    "request_prior_authorization",
    "escalate_to_human_specialist",
)

TEMPLATES = {
    "check_patient_policy": {
        "clean": [
            "Is {name}'s {treatment} covered, and what is the policy limit?",
            "Check the coverage status and limit for {name}'s {treatment}.",
            "Does {name}'s plan cover {treatment}? Please give the coverage limit.",
            "Look up whether {treatment} is covered for {name}.",
            "What does the policy say about {name}'s {treatment}?",
            "Can you verify coverage and the limit for {treatment} for {name}?",
            "Tell me if {name} has coverage for {treatment} and the maximum amount.",
            "Please check the policy rule for {name}'s {treatment}.",
        ],
        "paraphrase": [
            "I need to know what insurance will allow for {name}'s {treatment}.",
            "Would {name}'s plan pay toward {treatment}, and up to how much?",
            "Can you look into the benefits available to {name} for {treatment}?",
            "Find the policy details that apply to {name}'s {treatment}.",
            "How does the insurance handle {treatment} in {name}'s case?",
            "Please see what reimbursement terms apply to {name}'s {treatment}.",
            "Is there a benefit cap for {name} when it comes to {treatment}?",
            "I am checking {name}'s benefits for {treatment}; what does the plan allow?",
        ],
        "typo": [
            "Is {name}'s {treatment} coverd and what is the policy limt?",
            "Can you chek coverage for {name}'s {treatment}?",
            "Look up the insurence rule for {name}'s {treatment}.",
            "Does {name} have benifits for {treatment}?",
            "Please verfy the coverage limit for {name}'s {treatment}.",
            "Is the {treatment} aproved under {name}'s policy?",
            "What is the coverge status of {name}'s {treatment}?",
            "Check if {name}'s plan covrs {treatment}, please.",
        ],
        "ambiguous": [
            "Can you check {name}'s plan for me?",
            "What does the policy say about {name}?",
            "Can you look up whether this is included for {name}?",
            "Is {name}'s treatment covered?",
            "Please check the insurance details for {name}.",
            "What is the coverage situation for {name}'s care?",
            "Can you confirm what {name} is eligible for?",
            "Look up the plan limit for {name}, please.",
        ],
        "multi_intent": [
            "Check coverage for {name}'s {treatment}, then tell me the diagnosis on file.",
            "Is {name}'s {treatment} covered? Also confirm the policy limit.",
            "Look up {name}'s {treatment} coverage and check the treatment listed on file.",
            "Please verify the {treatment} rule for {name} and give me the coverage cap.",
            "Tell me if {name}'s {treatment} is covered and what diagnosis the record shows.",
            "Check the policy for {name}'s {treatment}, and confirm the patient's age.",
            "Find {name}'s coverage status for {treatment} and the applicable limit.",
            "First check whether {name}'s {treatment} is covered, then summarize the policy rule.",
        ],
    },
    "calculate_out_of_pocket": {
        "clean": [
            "If {name}'s treatment costs ${cost} and the limit is ${limit}, how much is out of pocket?",
            "Calculate what {name} owes on a ${cost} bill with ${limit} of coverage.",
            "How much remains for {name} to pay after ${limit} coverage on a ${cost} treatment?",
            "Work out {name}'s out-of-pocket amount for ${cost} in charges and a ${limit} limit.",
            "For {name}, subtract the ${limit} coverage limit from the ${cost} treatment cost.",
            "What balance does {name} owe if the treatment is ${cost} and insurance covers up to ${limit}?",
            "Compute {name}'s patient share for a ${cost} treatment with a ${limit} cap.",
            "Please calculate the remaining amount on {name}'s ${cost} bill after a ${limit} benefit.",
        ],
        "paraphrase": [
            "After insurance pays up to ${limit}, what bill is left for {name} from ${cost}?",
            "How much would {name} need to pay themselves for a ${cost} service with ${limit} covered?",
            "Can you figure out the uncovered balance on {name}'s ${cost} charge, limit ${limit}?",
            "What is the gap between {name}'s ${cost} treatment and a ${limit} plan allowance?",
            "Estimate the patient responsibility: {name}, ${cost} cost, ${limit} benefit.",
            "Once the ${limit} benefit is applied to ${cost}, what is {name}'s share?",
            "Tell me the remaining bill for {name}; charges ${cost}, insurance maximum ${limit}.",
            "What portion of the ${cost} would not be paid by a ${limit} limit for {name}?",
        ],
        "typo": [
            "Calcualte {name}'s out of pocket on ${cost} with a ${limit} limt.",
            "How much does {name} owe after ${limit} coverge on a ${cost} bill?",
            "Work out {name}'s remaing balance for ${cost} cost and ${limit} insurence.",
            "What is the out-of-poket amount for {name}: ${cost} cost, ${limit} limit?",
            "Subtrct ${limit} from {name}'s ${cost} treatment bill.",
            "Can you calulate {name}'s patient share on ${cost} when the cap is ${limit}?",
            "How much is un-covered for {name} if treatment is ${cost} and limit ${limit}?",
            "Figure {name}'s balence after a ${limit} allowance against ${cost}.",
        ],
        "ambiguous": [
            "How much does {name} owe?",
            "What is the remaining balance for {name}?",
            "Can you calculate {name}'s patient share on this bill?",
            "How much is left after insurance for {name}?",
            "What would {name}'s out-of-pocket amount be?",
            "Can you work out the uncovered portion of {name}'s care?",
            "How much does the plan leave for {name} to pay?",
            "Please calculate the remainder for {name}.",
        ],
        "multi_intent": [
            "Calculate {name}'s share for ${cost} with a ${limit} cap, then check the policy.",
            "What is left on {name}'s ${cost} bill after ${limit} coverage, and is that covered?",
            "Work out the ${cost} minus ${limit} and tell me whether {name} needs authorization.",
            "Find {name}'s remaining amount for ${cost}/${limit}, then escalate if anything is disputed.",
            "Calculate the balance for {name} at ${cost} with ${limit} insurance and state the difference.",
            "How much does {name} owe on ${cost} after a ${limit} benefit, and what is the plan limit?",
            "Compute {name}'s copay from ${cost} and ${limit}, then check the treatment record.",
            "Please find the out-of-pocket amount for {name} and confirm the ${limit} limit.",
        ],
    },
    "request_prior_authorization": {
        "clean": [
            "Submit a prior authorization request for {name}'s {treatment}.",
            "Please request prior approval for {treatment} for {name}.",
            "Start a prior authorization for {name} to receive {treatment}.",
            "File an authorization request for {name}'s {treatment}.",
            "Can you initiate pre-approval for {treatment} for {name}?",
            "Request insurance authorization before {name} has {treatment}.",
            "Open a prior approval request for {name}'s {treatment}.",
            "I need a prior authorization submitted for {treatment} under {name}.",
        ],
        "paraphrase": [
            "Get the insurer's permission in advance for {name}'s {treatment}.",
            "Arrange the required pre-clearance for {treatment} for {name}.",
            "Can you have the plan review {name}'s {treatment} before it happens?",
            "Please send {name}'s {treatment} for advance insurance review.",
            "We need approval on file before {name} receives {treatment}; start that process.",
            "Put in the pre-certification request for {name}'s {treatment}.",
            "Have the insurer authorize {treatment} ahead of time for {name}.",
            "Please get advance clearance for {name} to undergo {treatment}.",
        ],
        "typo": [
            "Submmit prior auth for {name}'s {treatment}.",
            "Request pre-aproval for {treatment} for {name}.",
            "Start the authorizaton process for {name}'s {treatment}.",
            "Can you file an authrization for {name} to get {treatment}?",
            "Please get insurence approval in advance for {name}'s {treatment}.",
            "Open a prior authorisation requst for {treatment} under {name}.",
            "We need pre-certifcation for {name}'s {treatment}.",
            "Get {name}'s {treatment} pre-approved by the plan, please.",
        ],
        "ambiguous": [
            "Can you get this approved for {name}?",
            "Please start the approval process for {name}.",
            "The plan needs to sign off before treatment; can you help {name}?",
            "Can you arrange the review that {name} needs first?",
            "Please get advance approval for {name}'s care.",
            "The clinic says {name} needs clearance before the procedure.",
            "Can you send {name}'s request to insurance for review?",
            "We need permission before moving forward with {name}.",
        ],
        "multi_intent": [
            "Request prior authorization for {name}'s {treatment}, then check its coverage limit.",
            "Submit pre-approval for {name}'s {treatment} and tell me the treatment on file.",
            "Start authorization for {treatment} for {name}, and calculate the patient cost.",
            "File a prior auth for {name}'s {treatment}; if it is disputed, send it to a specialist.",
            "Get advance approval for {name}'s {treatment} and verify the policy rule.",
            "Please request authorization for {treatment} and check whether {name}'s plan covers it.",
            "Open the authorization case for {name}, then confirm the diagnosis in their record.",
            "Have {name}'s {treatment} reviewed in advance and report the coverage limit.",
        ],
    },
    "escalate_to_human_specialist": {
        "clean": [
            "Escalate {name}'s disputed {treatment} claim to a human specialist.",
            "Please connect {name} with a specialist about this denied claim.",
            "A human needs to review {name}'s urgent insurance complaint.",
        ],
        "paraphrase": [
            "Have a person look into {name}'s unresolved claim issue.",
            "Arrange a call from an insurance specialist for {name}'s dispute.",
            "Please pass {name}'s complaint to someone who can review it.",
        ],
        "typo": [
            "Escalate {name}'s claim dispuit to a human specialist.",
            "Please get a real person to review {name}'s urgent complant.",
            "Have a specalist call {name} about this claim issue.",
        ],
        "ambiguous": [
            "Can someone help {name} with this issue?",
            "I need a real person to call {name} back.",
            "Please have a specialist review what happened to {name}.",
        ],
        "multi_intent": [
            "Escalate {name}'s disputed claim and ask a specialist to check the coverage decision.",
            "Please connect {name} to a human and review the urgent claim complaint.",
        ],
        "out_of_scope": [
            "This is outside the insurance tools: {name} needs a person to advise on a new symptom.",
            "I know this is not a policy question; get a human to help {name} choose a medication.",
            "Please send {name} to a live representative for a legal question about the accident.",
            "A human should handle {name}'s request to change a pharmacy, which this tool cannot do.",
            "This is not a coverage lookup; connect {name} with someone about finding a doctor.",
            "Can a person help {name} book an appointment? This assistant cannot schedule it.",
            "Route {name}'s request for medical advice to a qualified human; it is outside insurance support.",
            "This is beyond the tool's scope: {name} needs a person to update contact details.",
            "Please find a human for {name}'s request to cancel the plan; do not process it here.",
            "This is not a claim question. Have someone help {name} replace a lost insurance card.",
            "I need a real person to answer {name}'s question about medication side effects.",
            "Connect {name} to a human for help selecting an in-network doctor.",
            "The assistant cannot book transport; have a representative contact {name}.",
            "Please send {name}'s request for a specialist recommendation to a human.",
            "This is outside the available tools: {name} wants help changing their mailing address.",
            "A person should answer {name}'s question about a prescription, not the routing assistant.",
            "Have a live representative discuss plan enrollment with {name}; it is not a policy lookup.",
            "Please ask a human to help {name} schedule a routine checkup.",
            "This request is outside claim support. Connect {name} with someone about a bill payment plan.",
            "A specialist should handle {name}'s request for a copy of their medical records.",
            "I need a human to explain {name}'s diagnosis; that is outside the insurance tools.",
            "This is not a coverage question; have a person help {name} find a nearby clinic.",
            "Please route {name}'s request to update a dependent to a live support representative.",
            "The available tools cannot give medical advice; have a qualified person speak with {name}.",
            "Have someone call {name} about a password reset, which this insurance assistant cannot do.",
            "This is out of scope: {name} wants a human to explain how to submit a general complaint.",
        ],
    },
}

CATEGORIES = (
    "clean",
    "paraphrase",
    "typo",
    "ambiguous",
    "multi_intent",
    "out_of_scope",
)


def build_records():
    """Render templates with names and facts from the fixed-seed patient dataset."""
    patients, policy_rows = build_seed_data()
    policy_limits = {row[0].casefold(): row[2] for row in policy_rows}
    records = []
    patient_index = 0

    for tool in TOOLS:
        for category in CATEGORIES:
            templates = TEMPLATES[tool].get(category, [])
            for template in templates:
                patient = patients[patient_index % len(patients)]
                name, _, diagnosis, treatment, cost = patient
                limit = policy_limits.get(treatment.casefold(), 0)
                query = template.format(
                    name=name,
                    diagnosis=diagnosis,
                    treatment=treatment,
                    cost=f"{cost:,.2f}",
                    limit=f"{limit:,.2f}",
                )
                records.append(
                    {
                        "id": len(records) + 1,
                        "query": query,
                        "tool": tool,
                        "category": category,
                    }
                )
                patient_index += 1

    return records


def _write_jsonl(path, records):
    """Write one compact JSON object per line."""
    with open(path, "w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_dataset(output_path=OUTPUT_PATH):
    """Write the full dataset and fixed-seed, tool-stratified train/test splits."""
    records = build_records()
    rng = random.Random(SPLIT_SEED)
    train_records = []
    test_records = []

    for tool in TOOLS:
        tool_records = [record for record in records if record["tool"] == tool]
        rng.shuffle(tool_records)
        train_records.extend(tool_records[:28])
        test_records.extend(tool_records[28:])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    _write_jsonl(f"{output_path}/eval_queries.jsonl", records)
    _write_jsonl(f"{output_path}/eval_train.jsonl", train_records)
    _write_jsonl(f"{output_path}/eval_test.jsonl", test_records)
    return records, train_records, test_records


if __name__ == "__main__":
    all_records, train_records, test_records = write_dataset()
    print(
        f"Wrote {len(all_records)} queries "
        f"({len(train_records)} train, {len(test_records)} test). "
        "Review every row manually before evaluation."
    )
