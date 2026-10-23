from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from .tools import tools

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

def configure_agent(model_name: str = "qwen2.5:7b-instruct", temperature: float = 0):
    """Configure Qwen LLM and bind available tools."""
    llm = ChatOllama(model=model_name, temperature=temperature)
    return llm.bind_tools(tools)

def run_agent_loop(user_input: str, llm_tool=None):
    """Run the agent loop with the user question and return the final response."""
    if llm_tool is None:
        llm_tool = configure_agent()

    message = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    while True:
        response = llm_tool.invoke(message)
        message.append(response)

        if not response.tool_calls:
            print("Agent: ", response.content)
            return response.content

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"Agent chose tool {tool_name} with the argument {tool_args}")

            selected_tool = None
            for tool in tools:
                if tool.name == tool_name:
                    selected_tool = tool
                    break

            if selected_tool:
                result = selected_tool.invoke(tool_args)
                print(f"Tool result: {result}")
                message.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
