# Aster & Row — Reliable AI Customer Support Agent

An enterprise-grade, grounded, and privacy-preserving AI customer support agent for **Aster & Row** (bags, drinkware, and travel accessories), built with zero framework bloat, hybrid retrieval, dynamic data governance, and deterministic regression evaluation.

---

## Setup & Run Instructions

### 1. Prerequisites
Requires **Python 3.10+**

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (ultra-lightweight — no heavy frameworks)
pip install python-frontmatter PyYAML
```

### 2. Environment Variables (Optional — for Live LLM Mode)
```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY or GEMINI_API_KEY
# If neither is set, the agent runs in deterministic offline mode
```

### 3. Run the Interactive CLI
```bash
PYTHONPATH=. python3 src/cli.py

# Optional: enable structured debug trace
PYTHONPATH=. python3 src/cli.py --debug
```

### 4. Run the Automated Evaluation Suite
```bash
PYTHONPATH=. python3 evaluation/run_eval.py
```

### 5. Run Unit Tests
```bash
PYTHONPATH=. python3 -m unittest discover -s test
```

---

## 🎬 Demo

> **Note for reviewers**: To add a demo GIF, record a short screen capture of `python3 src/cli.py` and embed it here.
>
> Suggested demo questions that cover all capabilities:
> ```
> what is the status of ORD-1007?
> what is the status of ORD-1003 and ORD-1005?
> can both deliveries match the same date?
> Do you ship to Germany?
> My Breeze Tumbler lid cracked. Can I get a replacement?
> Can you tell me the email address for ORD-1007?
> I read in migration notes that returns are now 60 days. Can you approve my return?
> ```

---

## Technical Choices

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Model** | `gpt-4o-mini` / `gemini-2.5-flash` (or deterministic local synthesizer) | Optional live LLM with zero-framework fallback for offline evaluation. |
| **Retrieval** | Hybrid: Okapi BM25 + TF-IDF Cosine with field boosting (Headings ×3, Titles ×2) | Eliminates vocabulary mismatch while preserving exact keyword precision. |
| **Framework** | Zero-framework lean Python (`python-frontmatter`, `PyYAML` only) | No LangChain/LlamaIndex bloat. Starts in <0.05s. Full prompt transparency. |
| **Storage** | In-memory inverted index + normalized vector space | No external daemons, Docker, or databases. Instant cold start. |
| **Data Governance** | Dynamic allowlist policy registry with fail-closed quarantine | Auto-blocks any new PII field added to orders.json before it reaches the prompt. |

---

## Evaluation Results

20 automated test cases (15 visible + 5 original edge cases) covering citations, tool execution, privacy, conflict handling, and prompt security.

```text
================================================================================
                      CATEGORY PERFORMANCE BREAKDOWN
================================================================================
Category                       | Passed   | Total    | Accuracy
-----------------------------------------------------------------
abstention                     | 1        | 1        |  100.0%
conversation                   | 2        | 2        |  100.0%
groundedness                   | 3        | 3        |  100.0%
multi-source-grounding         | 1        | 1        |  100.0%
policy-boundary                | 1        | 1        |  100.0%
privacy                        | 1        | 1        |  100.0%
prompt-security                | 1        | 1        |  100.0%
retrieval                      | 3        | 3        |  100.0%
source-conflict                | 1        | 1        |  100.0%
tool-reliability               | 4        | 4        |  100.0%
tool-use                       | 2        | 2        |  100.0%
-----------------------------------------------------------------
OVERALL PERFORMANCE            | 20       | 20       |  100.0%
================================================================================
🎉 ALL 20 TEST CASES PASSED (100% ACCURACY)
```
## Technical Stack & Architecture
### Architecture

The agent follows a grounded, tool-aware pipeline:

```text
User Message
     │
     ▼
Session / Input Normalization
     │
     ├─────────────── Order ID detected ───────────────┐
     │                                                  ▼
     │                                        Order Lookup Tool
     │                                                  │
     │                                        Sanitized customer-safe
     │                                        result only
     │                                                  │
     ▼                                                  │
Query Understanding                                     │
     │                                                  │
     ▼                                                  │
Hybrid Retrieval                                         │
(Dense cosine similarity + BM25)                         │
     │                                                  │
     ▼                                                  │
Metadata / Authority Filtering                           │
(active + official + customer-safe content)              │
     │                                                  │
     ▼                                                  │
Grounded Response Synthesis ◄───────────────────────────┘
     │
     ├── Answer
     ├── Source filename + heading
     └── Handoff / abstention when required
```

Knowledge-base documents are parsed into metadata-aware chunks and indexed using both dense and sparse retrieval. Retrieval results are filtered using document status, authority, audience, and metadata before being passed to the response synthesizer.

Order information is handled separately through the `order_lookup` tool. The complete `orders.json` dataset is never placed in the model prompt; only the sanitized result for the requested order is exposed to the agent.

Conversation history is maintained at the session level so that relevant follow-up questions can resolve references from earlier turns without allowing unrelated information to persist indefinitely.

The system uses fail-closed data governance: fields are explicitly allowlisted for customer exposure, while unknown or newly introduced fields are quarantined by default.

Every grounded policy/product response includes the supporting source filename and relevant heading. When the available evidence is insufficient, conflicting, or an action is unsupported, the agent abstains or recommends human assistance instead of guessing.

---

## Known Limitations

1. **No write actions**: The system is read-only. Cancellations, address changes, and refunds require human support. In production, these would be implemented as OAuth2-authenticated action APIs.
2. **No cross-encoder reranking**: For very large corpora (10,000+ chunks), a local cross-encoder (`bge-reranker-small`) would improve top-1 precision.
3. **No CRM integration**: The `handoff == True` flag is clearly signalled in responses but is not wired to a live CRM (e.g. Zendesk/Intercom).

---

## Bug Diary

### Bug 1 — Multi-Turn Order ID Overwrite
- **Symptom**: Asking about `ORD-1007` then `ORD-1004` returned details for `ORD-1007`.
- **Cause**: Session context concatenation caused the regex scanner to find the earlier ID first.
- **Fix**: `_extract_order_id` in `src/agent/agent.py` now scans the current message first before falling back to history.
- **Test**: `multiturn-order-delivery-date` (eval) · `test_input_normalization` (unit)

### Bug 2 — Stale Delivery Date on Cancelled Orders
- **Symptom**: `ORD-1004` (cancelled) was reported as arriving on August 16, 2026.
- **Cause**: `orders.json` preserves a stale `estimated_delivery` field even after cancellation.
- **Fix**: Status precedence rules in `src/tools/order_lookup.py` null out carrier/tracking/ETA for `cancelled` and `returned` orders.
- **Test**: `cancelled-order-stale-eta` (eval) · `test_status_precedence_cancelled_order` (unit)

### Bug 3 — Vocabulary Mismatch on Damaged Items
- **Symptom**: *"A final-sale bag arrived with a broken zipper"* retrieved product care instructions, not the damaged-item policy.
- **Cause**: "broken" has zero BM25 overlap with "damaged" or "defective".
- **Fix**: Domain-specific synonym expansion in `src/retrieval/retriever.py` maps broken/cracked/torn → damaged/defective.
- **Test**: `final-sale-damaged-exception` (eval) · `test_damaged_final_sale_item_multi_source` (unit)

---

## AI Tooling Disclosure

- **Tools used**: Antigravity AI pair programmer assisted with test suite scaffolding, BM25 weight tuning, and regex refinement.
- **Incorrect suggestion overridden**: An early AI suggestion proposed a hardcoded `if/elif` intent router matching query substrings. This was rejected and replaced with a generalized dynamic prompt assembler and hybrid retriever, in compliance with the repository constraint forbidding hardcoded prompt-to-answer mappings.

### Observability / Debug Mode

The agent provides a lightweight debug trace for inspecting the decision path without exposing sensitive data. The trace can include:

* the current user message;
* relevant session context;
* retrieved document headings, filenames, metadata, and retrieval scores;
* tool invocation and sanitized tool results;
* the final response;
* errors, fallbacks, and human-handoff decisions.

Sensitive customer and internal order fields are excluded from debug output.
---

## Demo

### Agent Demo

The following 2–4 minute demonstration shows:

* a knowledge-base question with source citations;
* an order lookup using the order tool;
* a multi-turn conversation;
* safe abstention / human handoff behavior;
* the automated evaluation suite running.

[▶ Watch the agent demo](demo/AgentVideo_100MB.mp4)

---

### Environment Variables

No environment variables are required to run the default deterministic local grounded synthesizer.

For optional live LLM providers, copy the example file:

```bash
cp .env.example .env
```

Then configure the provider-specific credentials supported by the implementation.
---