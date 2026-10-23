from .compare_agent import process_user_query
from generate_plots import generate_benchmark_plots


def main():
    print("=== Healthcare Assistant Ready ===")
    print("Ask any healthcare question in plain English (type 'exit' or 'quit' to stop).\n")

    while True:
        try:
            user_input = input("You : ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nSession ended. Updating benchmark comparison plots...")
                generate_benchmark_plots()
                print("Goodbye!")
                break

            # The agent understands the query, picks the tool, and returns ONLY plain text
            response = process_user_query(user_input)
            print(f"\n{response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            generate_benchmark_plots()
            break


if __name__ == "__main__":
    main()
