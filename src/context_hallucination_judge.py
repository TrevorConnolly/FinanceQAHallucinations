"""
Context-grounded hallucination scoring for RAG (LLM-as-judge).

This operationalizes the definition used across RAG and attribution work: a
*contextual hallucination* is content in the model answer that is not entailed
by (or contradicts) the retrieved passages—the only evidence the model was
given. This is distinct from (a) answer correctness vs. a gold label, and
(b) invalid citation IDs, which this module does not judge.

Useful for A/B comparisons when you change retrieval, prompting, or reranking:
report mean(binary flag) as *context hallucination rate*, mean *claim_level_rate*
(|unsupported| / |atomic claims|), and optionally mean *severity*.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.0


def judge_context_hallucination(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """
    Returns a dict with:
      - has_context_hallucination: bool
      - hallucination_severity: float in [0, 1] (higher = more / worse)
      - atomic_claims: list[str] (distinct factual atomic claims in the answer)
      - total_claims: int (len(atomic_claims))
      - unsupported_claims: list[str] (subset not grounded in passages)
      - claim_level_hallucination_rate: float | None (unsupported/total; None if total==0)
    """
    cli = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    contexts_block = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{c}" for i, c in enumerate(contexts)
    )
    prompt = f"""You evaluate answers from a retrieval-augmented (RAG) system.

DEFINITION — CONTEXTUAL HALLUCINATION:
The answer contains at least one factual assertion that CANNOT be justified using ONLY the passages below, OR it CONTRADICTS those passages.
- Use ONLY the passages as evidence. Do not use outside or prior knowledge to mark claims as "true".
- Hedged uncertainty ("may", "appears to") tied to the passages is acceptable if not overstated.
- If the answer only states that information is missing (e.g. "Insufficient Information.") and adds no extra factual claims, use empty arrays for claims.

STEP 1 — List every distinct ATOMIC FACTUAL CLAIM in the ANSWER (short strings; one fact each). Omit non-factual boilerplate. If there are no factual claims, use an empty array.

STEP 2 — Among those atomic claims, list only the ones that CANNOT be justified from the passages or CONTRADICT them. Each entry must correspond to one atomic claim (same wording or clear paraphrase).

QUESTION:
{question}

PASSAGES (only source of truth for grounding):
{contexts_block}

ANSWER:
{answer}

Return a JSON object with exactly these keys:
- "atomic_claims": array of short strings (all atomic factual claims from the answer; [] if none)
- "unsupported_claims": array of short strings (unsupported/contradicted subset; [] if none)
- "has_context_hallucination": boolean (true iff unsupported_claims is non-empty)
- "hallucination_severity": number from 0.0 to 1.0 — use len(unsupported)/len(atomic) when atomic is non-empty, else 0.0; cap at 1.0
"""

    response = cli.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict grounding auditor. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=DEFAULT_TEMPERATURE,
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)

    atomic = data.get("atomic_claims") or []
    if not isinstance(atomic, list):
        atomic = []
    atomic = [str(c).strip() for c in atomic if str(c).strip()]

    unsupported = data.get("unsupported_claims") or []
    if not isinstance(unsupported, list):
        unsupported = []
    unsupported = [str(c).strip() for c in unsupported if str(c).strip()]

    total = len(atomic)
    if total == 0:
        claim_rate = None
        has_h = False
        unsupported = []
        sev = 0.0
    else:
        claim_rate = min(1.0, len(unsupported) / total)
        has_h = len(unsupported) > 0
        try:
            sev = float(data.get("hallucination_severity", claim_rate))
        except (TypeError, ValueError):
            sev = claim_rate
        sev = max(0.0, min(1.0, sev))

    return {
        "has_context_hallucination": has_h,
        "hallucination_severity": sev,
        "atomic_claims": atomic,
        "total_claims": total,
        "unsupported_claims": unsupported,
        "claim_level_hallucination_rate": claim_rate,
    }
