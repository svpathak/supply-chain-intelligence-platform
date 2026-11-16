"""Interactive command-line entry point for the Groq supply-chain assistant."""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from app.assistant.groq_assistant import GroqSupplyChainAssistant


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Supply Chain Intelligence assistant (Groq)")
    parser.add_argument("--question", help="Ask one question and exit; omit for interactive mode")
    args = parser.parse_args()
    assistant = GroqSupplyChainAssistant()
    if args.question:
        print(assistant.answer(args.question))
        return
    print("Supply Chain Intelligence assistant. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                print("\nAssistant> " + assistant.answer(question))
            except Exception as exc:
                print(f"\nAssistant error: {exc}")


if __name__ == "__main__":
    main()
