import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Dynamic root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import CustomerSupportAgent, AgentResponse

VISIBLE_CASES_PATH = PROJECT_ROOT / "evaluation" / "visible-cases.json"


# ============================================================================
# 5 ORIGINAL TEST CASES (Required by Assignment Rubric)
# ============================================================================
ORIGINAL_CASES = [
    {
        "id": "gift-card-return-refusal",
        "category": "policy-boundary",
        "messages": [
            {
                "role": "user",
                "content": "I received an unused $50 Aster & Row gift card as a gift. Can I return it for a cash refund?"
            }
        ],
        "expect": {
            "must_include_concepts": [
                "gift cards are final sale",
                "cannot be returned",
                "exchanged for cash"
            ],
            "must_not_include": [
                "30 calendar days"
            ],
            "required_sources": [
                "10-gift-cards-and-price-adjustments.md"
            ],
            "tool": "not_called",
            "handoff": False
        }
    },
    {
        "id": "price-adjustment-policy",
        "category": "retrieval",
        "messages": [
            {
                "role": "user",
                "content": "I bought the Metro Backpack 5 days ago and saw it just went on sale. Can I get a price adjustment?"
            }
        ],
        "expect": {
            "must_include_concepts": [
                "7 calendar days",
                "price adjustment"
            ],
            "required_sources": [
                "10-gift-cards-and-price-adjustments.md"
            ],
            "tool": "not_called",
            "handoff": False
        }
    },
    {
        "id": "multiturn-order-delivery-date",
        "category": "conversation",
        "messages": [
            {
                "role": "user",
                "content": "Can you check status for order ORD-1006?"
            },
            {
                "role": "user",
                "content": "When was it delivered?"
            }
        ],
        "expect": {
            "must_include_concepts": [
                "delivered",
                "August 10, 2026"
            ],
            "tool": "order_lookup",
            "tool_arguments": {
                "order_id": "ORD-1006"
            },
            "handoff": False
        }
    },
    {
        "id": "returned-order-stale-carrier-suppression",
        "category": "tool-reliability",
        "messages": [
            {
                "role": "user",
                "content": "What is the tracking number for returned order ORD-1008?"
            }
        ],
        "expect": {
            "must_include_concepts": [
                "returned"
            ],
            "must_not_include": [
                "1ZAR100800000008",
                "arriving"
            ],
            "tool": "order_lookup",
            "handoff": False
        }
    },
    {
        "id": "no-action-promise-warranty-claim",
        "category": "groundedness",
        "messages": [
            {
                "role": "user",
                "content": "My Breeze Tumbler lid cracked. Please process an immediate replacement right now."
            }
        ],
        "expect": {
            "must_include_concepts": [
                "warranty",
                "support"
            ],
            "must_not_include": [
                "replacement has been ordered",
                "replacement processed"
            ],
            "required_sources": [
                "07-warranty.md"
            ],
            "tool": "not_called",
            "handoff": True
        }
    }
]


def load_all_eval_cases() -> List[Dict[str, Any]]:
    """Loads visible cases and appends the 5 original test cases."""
    with open(VISIBLE_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_cases = list(data.get("cases", []))
    all_cases.extend(ORIGINAL_CASES)
    return all_cases


def evaluate_single_case(case: Dict[str, Any], agent: CustomerSupportAgent) -> Tuple[bool, List[str]]:
    """
    Executes all conversation turns in a test case and runs deterministic assertions.
    """
    case_id = case["id"]
    messages = case["messages"]
    expect = case.get("expect", {})
    session_id = f"eval_{case_id}"

    agent.reset_session(session_id)

    last_response: Optional[AgentResponse] = None
    all_responses: List[AgentResponse] = []

    # Execute all dialogue turns in sequence within the same session
    for msg in messages:
        if msg["role"] == "user":
            last_response = agent.process_message(msg["content"], session_id=session_id)
            all_responses.append(last_response)

    if not last_response:
        return False, ["No response generated by agent."]

    failures = []
    combined_answer = "\n".join(r.answer for r in all_responses).lower()

    # 1. Check 'must_include'
    for phrase in expect.get("must_include", []):
        if phrase.lower() not in combined_answer:
            failures.append(f"Missing required phrase: '{phrase}'")

    # 2. Check 'must_include_concepts'
    for concept in expect.get("must_include_concepts", []):
        keywords = [w.lower() for w in re.findall(r"\w+", concept) if len(w) > 2]
        matched_words = [w for w in keywords if w in combined_answer]
        if len(matched_words) < max(1, len(keywords) // 2):
            failures.append(f"Missing required concept: '{concept}'")

    # 3. Check 'must_not_include'
    for forbidden in expect.get("must_not_include", []):
        if forbidden.lower() in combined_answer:
            failures.append(f"Found forbidden text: '{forbidden}'")

    # 4. Check 'must_not_invent'
    for invented in expect.get("must_not_invent", []):
        if invented.lower() in combined_answer:
            failures.append(f"Agent invented unsupported field: '{invented}'")

    # 5. Check 'required_sources'
    cited_sources = set()
    for r in all_responses:
        cited_sources.update(r.sources)
        for s in re.findall(r"\b(\d{2}-[a-z0-9\-]+\.md)\b", r.answer):
            cited_sources.add(s)

    for req_src in expect.get("required_sources", []):
        if req_src not in cited_sources:
            failures.append(f"Missing required source citation: '{req_src}'")

    # 6. Check 'forbidden_sources_as_authority'
    for forb_src in expect.get("forbidden_sources_as_authority", []):
        if forb_src in cited_sources:
            failures.append(f"Forbidden source cited as authority: '{forb_src}'")

    # 7. Check Tool Behavior
    expected_tool = expect.get("tool")
    if expected_tool:
        actual_tools = [r.tool for r in all_responses]
        if expected_tool == "not_called":
            if any(t != "not_called" for t in actual_tools):
                failures.append(f"Tool was called when expected 'not_called': {actual_tools}")
        elif expected_tool == "not_called_without_id":
            if any(t == "order_lookup" for t in actual_tools):
                failures.append("Order lookup tool was executed without an order ID.")
        elif expected_tool == "order_lookup":
            if "order_lookup" not in actual_tools:
                failures.append(f"Expected tool 'order_lookup' was not called. Actual: {actual_tools}")
        elif expected_tool == "optional_sanitized_lookup":
            pass

    # 8. Check Tool Arguments
    expected_tool_args = expect.get("tool_arguments")
    if expected_tool_args:
        actual_args = [r.tool_arguments for r in all_responses if r.tool_arguments]
        if not any(a == expected_tool_args for a in actual_args):
            failures.append(f"Tool arguments mismatch. Expected: {expected_tool_args}, Actual: {actual_args}")

    # 9. Check Handoff
    expected_handoff = expect.get("handoff")
    if expected_handoff is not None:
        actual_handoff = any(r.handoff for r in all_responses)
        if actual_handoff != expected_handoff:
            failures.append(f"Handoff mismatch: expected {expected_handoff}, got {actual_handoff}")

    passed = len(failures) == 0
    return passed, failures


def run_evaluation():
    """
    Main evaluation runner:
    Executes all cases, tabulates category scores, and reports benchmark summary.
    """
    print("=" * 80)
    print("     ASTER & ROW AI SUPPORT AGENT — AUTOMATED EVALUATION SUITE")
    print("=" * 80)

    agent = CustomerSupportAgent()
    cases = load_all_eval_cases()

    category_stats: Dict[str, Dict[str, int]] = {}
    passed_count = 0
    failed_count = 0

    print(f"\nRunning {len(cases)} test cases ({len(cases) - 5} visible cases + 5 original edge cases)...\n")

    for case in cases:
        cid = case["id"]
        category = case.get("category", "general")
        passed, failures = evaluate_single_case(case, agent)

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1

        if passed:
            passed_count += 1
            category_stats[category]["passed"] += 1
            print(f"  ✅ [PASS] {cid:<42} ({category})")
        else:
            failed_count += 1
            print(f"  ❌ [FAIL] {cid:<42} ({category})")
            for f in failures:
                print(f"       └─ {f}")

    print("\n" + "=" * 80)
    print("                      CATEGORY PERFORMANCE BREAKDOWN")
    print("=" * 80)
    print(f"{'Category':<30} | {'Passed':<8} | {'Total':<8} | {'Accuracy':<10}")
    print("-" * 65)

    for cat, stats in sorted(category_stats.items()):
        acc = (stats["passed"] / stats["total"]) * 100
        print(f"{cat:<30} | {stats['passed']:<8} | {stats['total']:<8} | {acc:>6.1f}%")

    print("-" * 65)
    overall_acc = (passed_count / len(cases)) * 100
    print(f"{'OVERALL PERFORMANCE':<30} | {passed_count:<8} | {len(cases):<8} | {overall_acc:>6.1f}%")
    print("=" * 80)

    if failed_count == 0:
        print("\n🎉 ALL 20 TEST CASES PASSED (100% ACCURACY)!\n")
    else:
        print(f"\n⚠️ {failed_count} test case(s) failed. See logs above.\n")


if __name__ == "__main__":
    run_evaluation()

