import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import Tool, tool
import psycopg2

load_dotenv()

connection = psycopg2.connect(
    database=os.getenv("DB_NAME", "healthcare_agent"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432")
)

@tool
def patient_info(patient_name:str):
    """get patient details"""
    cursor=connection.cursor()
    query=" Select name, age, diagnosis, treatment from patients where lower(name)= lower(%s)"
    cursor.execute(query,(patient_name,))
    patient=cursor.fetchone()
    cursor.close()

    if patient is None:
        return f"No patient named {patient_name} found."

    return {
        'name': patient[0],
        "age":patient[1],
        'diagnosis': patient[2],
        'treatment':patient[3]
    }

@tool
def policy_coverage(treatment:str):
    """check whether the treatment is covered by the policy"""
    covered_treatment=["appendectomy", "chemotherapy","radiation therapy"]

    if treatment.lower() in covered_treatment:
        return f"{treatment} covered by the policy"

    return f"{treatment} not covered by the policy"

@tool
def coverage_limit(treatment:str):
    """Get the maximum amount covered for a treatment."""
    limit={
        "appendectomy":5000,
        "chemotherapy":30000,
        "radiation therapy":10000
    }
    if treatment.lower() in limit:
        return limit[treatment]
    return 0

@tool
def calculate_remaining(treatment:str,treatment_cost:float,coverage_limit:float):
    """calculate how much patient has to pay"""
    remaining=treatment_cost-coverage_limit
    if remaining>0:
        return f"patient has to pay {remaining}"
    return f"the treatment is covered by the policy"

tools=[patient_info,policy_coverage, coverage_limit, calculate_remaining]

llm=ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

llm_tool=llm.bind_tools(tools)

system_prompt = """
You are a healthcare insurance assistant.

Rules:
1. Always use tools to get patient information.
2. Never invent patient details.
3. Never invent treatment names.
4. Never invent insurance coverage or coverage limits.
5. Use database information as the source of truth.
6. If information is missing, say that it is missing.
7. Use the patient's actual treatment returned by the database.
"""

user_input=input("You : ")
message=[HumanMessage(content=user_input), SystemMessage(content=system_prompt)]

while True:
    response=llm_tool.invoke(message)
    message.append(response)

    if not response.tool_calls:
        print("Agent ", response.content)
        break

    for tool_call in response.tool_calls:
        tool_name=tool_call["name"]
        tool_args=tool_call["args"]
        print(f"Agent chose tool {tool_name} with the argument {tool_args}")

        selected_tool=None

        for tool in tools:
            if tool.name==tool_name:
                selected_tool=tool
                break

        if selected_tool:
            result=selected_tool.invoke(tool_args)

            message.append(ToolMessage(content=str(result),tool_call_id=tool_call["id"]))
