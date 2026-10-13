import sys
import os

# Ensure modules resolve whether executed from root or inside app
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from app.agent import configure_agent, run_agent_loop
except ImportError:
    from agent import configure_agent, run_agent_loop

def main():
    print("=== Healthcare Insurance Agent ===")
    print("Initializing agent...")
    agent = configure_agent()
    print("Agent ready! Ask a question (type 'exit' or 'quit' to stop).\n")

    while True:
        try:
            user_input = input("You : ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                print("Exiting healthcare assistant. Goodbye!")
                break

            run_agent_loop(user_input, llm_tool=agent)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting healthcare assistant. Goodbye!")
            break

if __name__ == "__main__":
    main()
