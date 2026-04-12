"""
Named RAG experiment presets for A/B runs (chunking, top_k, prompts, strict citations).

Use with run_pipeline.py --experiment <id> and compare outputs under outputs/logs/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

# --- Header splitting presets (MarkdownHeaderTextSplitter) ---
HEADER_PRESETS: dict[str, list[tuple[str, str]]] = {
    "standard": [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ],
    "deep": [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ],
}


@dataclass(frozen=True)
class RAGExperimentConfig:
    """Immutable config for one experimental condition."""

    experiment_id: str
    top_k: int = 5
    prompt_preset: str = "default"
    strict_citation_postprocess: bool = False
    chunk_size: int = 1000
    chunk_overlap: int = 250
    header_preset: str = "standard"

    def header_levels(self) -> list[tuple[str, str]]:
        if self.header_preset not in HEADER_PRESETS:
            raise ValueError(
                f"Unknown header_preset {self.header_preset!r}; "
                f"choose from {list(HEADER_PRESETS)}"
            )
        return HEADER_PRESETS[self.header_preset]


_BASELINE = RAGExperimentConfig(
    experiment_id="baseline",
    top_k=5,
    prompt_preset="default",
    strict_citation_postprocess=False,
    chunk_size=1000,
    chunk_overlap=250,
    header_preset="standard",
)

# Register presets (add more with dataclasses.replace).
EXPERIMENTS: dict[str, RAGExperimentConfig] = {
    "baseline": _BASELINE,
    "k3": replace(_BASELINE, experiment_id="k3", top_k=3),
    "k8": replace(_BASELINE, experiment_id="k8", top_k=8),
    "k10": replace(_BASELINE, experiment_id="k10", top_k=10),
    "baseline_cot": replace(
        _BASELINE, experiment_id="baseline_cot", prompt_preset="chain_of_thought"
    ),
    "baseline_strict_cite": replace(
        _BASELINE,
        experiment_id="baseline_strict_cite",
        strict_citation_postprocess=True,
    ),
    "k8_cot": replace(
        _BASELINE,
        experiment_id="k8_cot",
        top_k=8,
        prompt_preset="chain_of_thought",
    ),
    "fine_chunks": replace(
        _BASELINE, experiment_id="fine_chunks", chunk_size=600, chunk_overlap=120
    ),
    "coarse_chunks": replace(
        _BASELINE, experiment_id="coarse_chunks", chunk_size=1500, chunk_overlap=300
    ),
    "deep_headers_fine": replace(
        _BASELINE,
        experiment_id="deep_headers_fine",
        header_preset="deep",
        chunk_size=700,
        chunk_overlap=140,
    ),
}


def get_experiment(experiment_id: str) -> RAGExperimentConfig:
    if experiment_id not in EXPERIMENTS:
        raise KeyError(
            f"Unknown experiment {experiment_id!r}. "
            f"Registered: {sorted(EXPERIMENTS)}"
        )
    return EXPERIMENTS[experiment_id]


def list_experiment_ids() -> list[str]:
    return sorted(EXPERIMENTS)


# --- Prompts (must stay aligned with citation format in generator) ---
_PROMPT_DEFAULT = """
You are an expert financial auditor. Answer the user's question using ONLY the provided context.

CRITICAL RULES:
1. Every single factual claim you make MUST be immediately followed by a citation to the specific chunk ID.
2. Format citations exactly like this: [Source: <chunk_id>].
3. Do not combine citations. If a sentence uses two sources, cite both separately.
4. If the provided context does not contain the answer, output exactly: "Insufficient Information."
5. Do not make up outside information or use prior knowledge.
""".strip()


_PROMPT_CHAIN_OF_THOUGHT = """
You are an expert financial auditor. Answer the user's question using ONLY the provided context.

WORKFLOW (keep this brief in your reply — then give the final cited answer):
1. Note which chunk IDs are relevant to the question.
2. List the specific facts you will use from those chunks (no facts without a chunk).
3. Write the final answer: every factual claim must be immediately followed by [Source: <chunk_id>].

CRITICAL RULES:
1. Every single factual claim you make MUST be immediately followed by a citation to the specific chunk ID.
2. Format citations exactly like this: [Source: <chunk_id>].
3. Do not combine citations. If a sentence uses two sources, cite both separately.
4. If the provided context does not contain the answer, output exactly: "Insufficient Information."
5. Do not make up outside information or use prior knowledge.
""".strip()

PROMPT_PRESETS: dict[str, str] = {
    "default": _PROMPT_DEFAULT,
    "chain_of_thought": _PROMPT_CHAIN_OF_THOUGHT,
}


def build_system_prompt(preset: str) -> str:
    if preset not in PROMPT_PRESETS:
        raise ValueError(
            f"Unknown prompt_preset {preset!r}; choose from {list(PROMPT_PRESETS)}"
        )
    return PROMPT_PRESETS[preset]


def save_experiment_snapshot(
    config: RAGExperimentConfig,
    logs_dir: str,
    corpus_id: str | None = None,
) -> Path:
    """Write JSON next to results for reproducibility."""
    prefix = f"{corpus_id}_" if corpus_id else ""
    path = Path(logs_dir) / f"{prefix}{config.experiment_id}_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(config)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
