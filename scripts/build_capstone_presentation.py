#!/usr/bin/env python3
"""Generate capstone PowerPoint (python-pptx). Run: python3 scripts/build_capstone_presentation.py"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "presentation"
OUT.mkdir(parents=True, exist_ok=True)


def _bullets(slide, lines: list[str]) -> None:
    body = slide.placeholders[1]
    tf = body.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(18)


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    # --- 1. Title ---
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = (
        "Comparing RAG System Designs for Hallucination Reduction\n"
        "in Finance and Healthcare Documents"
    )
    st = slide.placeholders[1]
    st.text = (
        "[Your Name]\n"
        "[Course / Class]\n"
        "Adviser: [Adviser Name]\n"
        "\n"
        "Finance QA Hallucinations — Capstone Presentation"
    )

    # --- 2. Motivation & Goal ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Motivation and Goal"
    _bullets(
        slide,
        [
            "Topic: Reliable deployment of AI in professional workflows where answers must be precise (finance, healthcare).",
            "Problem: Hallucinations remain the main barrier to trust—especially with sensitive or regulated information.",
            'Goal: "The goal of my project is to empirically compare RAG system designs and measure which configurations best reduce hallucinations and improve answer quality—across document types representative of finance (10-Ks) and healthcare (FDA drug labels)."',
            "Why it matters: Almost every enterprise is adopting AI for productivity; reliability determines whether professionals actually use it.",
            "Audience benefit: Teams can prioritize retrieval depth, prompting, citation enforcement, and chunking strategies grounded in measured tradeoffs—not guesswork.",
        ],
    )

    # --- 3. Industry motivation (EY) ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Why This Problem Is Urgent (Industry Perspective)"
    _bullets(
        slide,
        [
            "Internship (EY): Clients consistently rank hallucinations and reliability as top concerns for AI adoption on sensitive data.",
            "Interviews with EY tech risk partners: among customer complaints, hallucinations and trust dominate—while benefits (efficiency) are widely understood.",
            "Bridge from practice to research: measure hallucination-related failure modes under controlled RAG settings so practitioners can reduce risk while capturing efficiency gains.",
        ],
    )

    # --- 4. Related work ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Problem Background and Related Work"
    _bullets(
        slide,
        [
            "Consensus: LLMs can produce fluent but unsupported text; RAG grounds answers in retrieved evidence but does not eliminate hallucinations (retrieval errors, context misuse, parametric knowledge).",
            "Surveys & reviews: Recent surveys synthesize hallucination taxonomies, detection (faithfulness judges, consistency checks), and mitigation (RAG, prompting, constrained decoding). Examples: comprehensive LLM hallucination surveys on arXiv; RAG-focused mitigation reviews (e.g., MDPI / similar).",
            "Commercial reality: Vendors offer RAG stacks, guardrails, and evaluation—yet optimal design is domain- and corpus-dependent; generic defaults underperform.",
            "Gap: Less clear guidance on which RAG knobs (top-k, CoT, strict citations, chunking) trade off grounding vs correctness for regulatory-style prose vs financial filings—this project tests that empirically on real PDF corpora.",
        ],
    )

    # --- 5. Approach ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Approach (Key Idea)"
    _bullets(
        slide,
        [
            "Key idea: Hold evaluation methodology fixed; systematically vary RAG design (retrieval breadth, prompts, citation post-processing, chunking presets).",
            "Compare outcomes on two genres: SEC-style 10-Ks (Nvidia, JPM) vs FDA drug labels (Humira, Keytruda)—same pipeline, different language and structure.",
            "Unique angle: Not only which prompt wins globally, but whether improvements are consistent across industries and where tradeoffs appear (grounding vs correctness).",
            "Tier A vs Tier B: Tier A = same chunks/gold (fair A/B for top-k & prompts). Tier B = alternate chunking (separate gold)—reported separately to avoid apples-to-oranges comparisons.",
        ],
    )

    # --- 6. Implementation ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Implementation"
    _bullets(
        slide,
        [
            "Ingestion: LlamaParse PDF-to-Markdown; semantic chunking (LangChain splitters) with experiment-specific chunk/header presets.",
            "Retrieval & generation: Chroma + embeddings; BM25; cross-encoder reranker (BGE); OpenAI GPT-4o generator with chunk-ID citations.",
            "Experiments: baseline (k=5), k=3/8/10, chain-of-thought prompts, strict citation post-processing (force abstain if cites invalid/missing), and chunk ablations.",
            "Gold data: Synthetic Q/A from sampled chunks + critique filter; evaluation n=10 questions per run (pilot scale).",
            "Metrics: Rule-based (recall@k, citation issues), dedicated LLM judge for context-grounded hallucinations & claim rates, RAGAS (faithfulness, relevancy, correctness).",
            "Hardening: Rate-limit retries on OpenAI 429s for long batch runs.",
        ],
    )

    # --- 7. Evaluation design ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Evaluation Design"
    _bullets(
        slide,
        [
            "Corpora: 4 PDFs — 2 large 10-Ks, 2 long FDA labels (public regulatory text).",
            "Tier A: 7 presets vs shared baseline chunking — delta vs baseline per corpus, then average across corpora.",
            "Tier B: 3 chunking presets — separate gold per preset; compare within Tier B and genre pools.",
            "Success measures: Lower hallucination indicators where appropriate; higher answer correctness and retrieval of gold evidence; explicit tradeoff stories (e.g., correctness up, grounding unchanged).",
            "Caveat: n=10 is adequate for directional patterns and engineering decisions—not for publication-grade significance testing without more data.",
        ],
    )

    # --- 8. Results — Tier A cross-corpus ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Results — Tier A (Same Gold): Cross-Corpus Averages vs Baseline"
    _bullets(
        slide,
        [
            "Largest average reduction in context-grounded hallucination flag vs baseline: k=3 (mean delta roughly -0.08 across four corpora).",
            "Largest average gain in answer correctness vs baseline: k8 + CoT (mean delta roughly +0.03)—but also the largest average increase in the hallucination flag vs baseline (~+0.03)—clear tradeoff.",
            "Strict-cite post-processing: near-zero average change in context hallucination flag vs baseline in this aggregate; average answer correctness slightly down vs baseline—suggests more abstention/stricter answers at small n.",
            "Chain-of-thought (baseline_cot): mixed—small average correctness dip with some runs showing higher faithfulness in finance pool (check per-corpus variance).",
        ],
    )

    # --- 9. Results — by industry ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Results — Pooled by Domain (Tier A Means)"
    _bullets(
        slide,
        [
            "Finance (2x 10-Ks), pooled: higher recall@5 (~0.85 baseline) vs FDA labels (~0.65 baseline)—retrieval finds gold chunks more often on these filings in this setup.",
            "Healthcare labels (2x FDA), pooled: baseline context-hallucination flag ~0.10 mean vs finance ~0.15 (lower is better)—genre differences appear in aggregate (still small n).",
            "k=3 on labels: context flag 0.0 pooled vs baseline 0.10—promising directionally; verify stability with larger n.",
            "Takeaway: No single winner for both industries; best preset depends on metric priority (grounding vs correctness vs abstention).",
        ],
    )

    # --- 10. Tier B note ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Chunking Presets (Tier B) — Reported Separately"
    _bullets(
        slide,
        [
            "Fine / coarse / deep-header chunking each regenerates gold—questions differ from Tier A.",
            "Use Tier B to compare chunking strategies within the same genre pools, not as delta-vs-baseline from Tier A.",
            "Observed pattern (pooled by genre): finance favors some chunk presets on correctness in this run; label corpora more sensitive—warrants larger-n follow-up.",
        ],
    )

    # --- 11. Limitations ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Limitations and Next Steps"
    _bullets(
        slide,
        [
            "Small evaluated n (10) per experiment—directional, not definitive.",
            "LLM judges and RAGAS add cost and variance; metrics correlate but measure different failure modes.",
            "Four documents do not represent all of finance or healthcare—findings are illustrative for regulatory prose vs 10-K structure.",
            "Next: increase n, add confidence intervals / bootstrap, test additional rerankers and abstention policies, document-type-specific chunk defaults.",
        ],
    )

    # --- 12. Conclusion ---
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Conclusion"
    _bullets(
        slide,
        [
            "Hallucination-aware RAG is not one-size-fits-all: retrieval depth, prompting, and citation enforcement shift grounding and correctness differently.",
            "Empirical comparison on real 10-K and FDA-label text shows measurable genre effects and clear tradeoffs (e.g., k8+CoT vs k3).",
            "Practical impact: gives engineers a reproducible evaluation harness and evidence-backed starting points for finance vs healthcare-style documents.",
            "Thank you — questions?",
        ],
    )

    path = OUT / "RAG_Hallucination_Capstone.pptx"
    prs.save(path)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
