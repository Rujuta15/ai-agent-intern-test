import sys
import json
import argparse
from pathlib import Path

# Dynamic root resolution so imports work from any execution directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import CustomerSupportAgent


def start_interactive_cli(debug_mode: bool = False):
    print("=" * 75, flush=True)
    print("           ASTER & ROW — AI CUSTOMER SUPPORT AGENT (CLI)", flush=True)
    if debug_mode:
        print("                 [DEBUG / OBSERVABILITY MODE ACTIVE]", flush=True)
    print("=" * 75, flush=True)
    print("Commands: 'exit' to quit, '/debug' to toggle live observability trace.\n", flush=True)

    agent = CustomerSupportAgent()
    session_id = "cli_session"

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThank you for chatting with Aster & Row support. Goodbye!", flush=True)
                break
            if user_input.lower() == "/debug":
                debug_mode = not debug_mode
                status_str = "ENABLED" if debug_mode else "DISABLED"
                print(f"\n[Observability Trace {status_str}]\n", flush=True)
                continue

            response = agent.process_message(user_input, session_id=session_id)

            print(f"\nAgent:\n{response.answer}", flush=True)
            if response.sources:
                print(f"\n[Cited Sources: {', '.join(response.sources)}]", flush=True)
            if response.tool != "not_called":
                print(f"[Tool Used: {response.tool} (Args: {response.tool_arguments})]", flush=True)
            if response.handoff:
                print("[⚠️ Human Support Handoff Recommended]", flush=True)

            # Basic Observability & Debug Trace (Requirement 6)
            if debug_mode:
                trace_payload = {
                    "user_message": user_input,
                    "session_id": session_id,
                    "turn": response.debug_trace.get("turn"),
                    "effective_query": response.debug_trace.get("effective_query"),
                    "resolved_order_ids": response.debug_trace.get("resolved_order_ids"),
                    "retrieved_chunks": response.debug_trace.get("retrieved_chunks", []),
                    "tool_call": response.debug_trace.get("tool_call"),
                    "tool_result": response.debug_trace.get("tool_result"),
                    "handoff": response.handoff,
                    "conversation_history_length": len(agent.get_session_history(session_id))
                }
                print("\n" + "-" * 30 + " [DEBUG TRACE LOG] " + "-" * 30, flush=True)
                print(json.dumps(trace_payload, indent=2), flush=True)
                print("-" * 75, flush=True)

            print("-" * 75 + "\n", flush=True)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Goodbye!", flush=True)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aster & Row AI Support Agent CLI")
    parser.add_argument("--debug", action="store_true", help="Enable structured observability debug trace")
    args = parser.parse_args()
    start_interactive_cli(debug_mode=args.debug)
