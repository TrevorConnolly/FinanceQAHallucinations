import json
import os
import re
import pandas as pd
from datasets import Dataset
from ragas import RunConfig, evaluate
from openai import OpenAI
from ragas.llms import llm_factory
from generator import AuditRAGGenerator
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._answer_correctness import answer_correctness
from context_hallucination_judge import judge_context_hallucination
from experiment_config import RAGExperimentConfig, get_experiment, save_experiment_snapshot

# --- CONFIGURATION ---
GOLD_DATASET_PATH = "data/synthetic/gold_dataset.csv"
OUTPUT_RESULTS_PATH = "outputs/logs/baseline_results.csv"
LOGS_DIR = "outputs/logs"

# ⚠️ Change this to None when you are ready to run all gold-dataset questions!
TEST_LIMIT = 3

# RAGAS judge: higher max_tokens avoids truncated structured JSON; longer timeout avoids TimeoutError
RAGAS_JUDGE_MODEL = "gpt-4o"
RAGAS_JUDGE_MAX_TOKENS = 8192
RAGAS_JUDGE_TEMPERATURE = 0.0
RAGAS_METRIC_TIMEOUT_SEC = 600
RAGAS_MAX_WORKERS = 8

# Context-grounded hallucination: LLM judges whether the answer adds facts
# not supported by retrieved passages (see context_hallucination_judge.py).
RUN_CONTEXT_HALLUCINATION_JUDGE = True

# Must match generator.AuditRAGGenerator citation format
CITATION_PATTERN = re.compile(r"\[Source:\s*(.*?)\]", re.DOTALL)


def extract_cited_chunk_ids(answer: str) -> list[str]:
    """UUIDs cited in the model answer (same delimiter as the generator)."""
    if not isinstance(answer, str):
        return []
    return [m.strip() for m in CITATION_PATTERN.findall(answer) if m.strip()]


def run_evaluation_pipeline(
    test_limit: int | None = None,
    evaluate_all: bool = False,
    *,
    experiment: RAGExperimentConfig | None = None,
    output_results_path: str | None = None,
) -> None:
    """
    Run RAG + RAGAS on the gold CSV.

    Default: use ``TEST_LIMIT`` from this module (may be None for full CSV).
    ``test_limit=N``: evaluate the first N rows.
    ``evaluate_all=True``: evaluate every row (ignores ``TEST_LIMIT`` and ``test_limit``).

    ``experiment``: RAG preset; results go to ``outputs/logs/{experiment_id}_results.csv``
    unless ``output_results_path`` is set.
    """
    exp = experiment or get_experiment("baseline")
    out_csv = output_results_path or os.path.join(
        LOGS_DIR, f"{exp.experiment_id}_results.csv"
    )
    save_experiment_snapshot(exp, LOGS_DIR)

    print(
        f"Initializing Evaluation Pipeline (experiment={exp.experiment_id})..."
    )
    print(f"   -> Results CSV: {out_csv}")

    if evaluate_all:
        limit = None
    elif test_limit is not None:
        limit = test_limit
    else:
        limit = TEST_LIMIT

    df_gold = pd.read_csv(GOLD_DATASET_PATH)
    if limit is not None:
        df_gold = df_gold.head(limit)
        print(f"   -> Evaluating {len(df_gold)} question(s) (limit={limit}).")
    else:
        print(f"   -> Evaluating all {len(df_gold)} question(s) in gold dataset.")

    # 2. Initialize the Student (Generator)
    generator = AuditRAGGenerator(experiment=exp)
    judge_client = (
        OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if RUN_CONTEXT_HALLUCINATION_JUDGE
        else None
    )

    # 3. Execution Loop: Run the RAG system on every question
    results = []
    print("\n🏃‍♂️ Running Generator on Test Set...")
    
    for index, row in df_gold.iterrows():
        print(f"   Processing Q{index + 1}/{len(df_gold)}...")
        
        # Run the RAG pipeline (top_k from experiment unless you extend this call)
        rag_output = generator.answer_question(row["question"])
        
        # Calculate Recall@5 (Did we find the Gold Chunk?)
        gold_id = str(row["gold_chunk_id"]).strip()
        recall_success = 1 if gold_id in rag_output["retrieved_ids"] else 0

        # Calculate Citation Hallucinations (Did it cite a fake ID?)
        citation_hallucination = 1 if len(rag_output["citation_errors"]) > 0 else 0

        # Citation vs gold (no LLM): did the answer cite the gold-standard chunk at least once?
        cited_ids = extract_cited_chunk_ids(rag_output["answer"])
        gold_chunk_cited = 1 if gold_id in cited_ids else 0

        ctx_hall_bin = float("nan")
        ctx_hall_sev = float("nan")
        ctx_hall_claims = "[]"
        ctx_atomic_json = "[]"
        ctx_total_claims = float("nan")
        ctx_unsupported_n = float("nan")
        ctx_claim_rate = float("nan")
        if judge_client is not None:
            try:
                ch = judge_context_hallucination(
                    row["question"],
                    rag_output["answer"],
                    rag_output["retrieved_contexts"],
                    client=judge_client,
                    model=RAGAS_JUDGE_MODEL,
                )
                ctx_hall_bin = 1.0 if ch["has_context_hallucination"] else 0.0
                ctx_hall_sev = float(ch["hallucination_severity"])
                uclaims = ch["unsupported_claims"]
                ctx_hall_claims = json.dumps(uclaims)
                ctx_atomic_json = json.dumps(ch["atomic_claims"])
                ctx_total_claims = float(ch["total_claims"])
                ctx_unsupported_n = float(len(uclaims))
                cr = ch["claim_level_hallucination_rate"]
                ctx_claim_rate = float("nan") if cr is None else float(cr)
            except Exception as e:
                print(
                    f"   Warning: context hallucination judge failed for Q{index + 1}: {e}"
                )

        # Store all data; RAGAS component scores merged after evaluate()
        results.append(
            {
                "experiment_id": exp.experiment_id,
                "exp_top_k": exp.top_k,
                "exp_prompt_preset": exp.prompt_preset,
                "exp_strict_citation_postprocess": int(exp.strict_citation_postprocess),
                "exp_chunk_size": exp.chunk_size,
                "exp_chunk_overlap": exp.chunk_overlap,
                "exp_header_preset": exp.header_preset,
                "question": row["question"],
                "ground_truth": row["ground_truth_answer"],
                "generated_answer": rag_output["answer"],
                "contexts": rag_output["retrieved_contexts"],
                "gold_chunk_id": row["gold_chunk_id"],
                "retrieved_ids": rag_output["retrieved_ids"],
                "cited_chunk_ids": json.dumps(cited_ids),
                "recall_at_5": recall_success,
                "citation_hallucination": citation_hallucination,
                "citation_issue_pre_strict": int(
                    bool(rag_output.get("citation_issue_pre_strict"))
                ),
                "strict_abstention_applied": int(
                    bool(rag_output.get("strict_abstention_applied"))
                ),
                "gold_chunk_cited": gold_chunk_cited,
                "context_hallucination": ctx_hall_bin,
                "context_hallucination_severity": ctx_hall_sev,
                "total_claims": ctx_total_claims,
                "unsupported_claim_count": ctx_unsupported_n,
                "claim_level_hallucination_rate": ctx_claim_rate,
                "atomic_claims_json": ctx_atomic_json,
                "unsupported_claims_json": ctx_hall_claims,
            }
        )
        
    df_results = pd.DataFrame(results)
    
    # 4. RAGAS Evaluation (LLM-as-a-Judge)
    print(
        f"\nRunning RAGAS evaluation "
        f"(judge={RAGAS_JUDGE_MODEL}, max_tokens={RAGAS_JUDGE_MAX_TOKENS}, "
        f"metric_timeout={RAGAS_METRIC_TIMEOUT_SEC}s)..."
    )
    
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory(
        RAGAS_JUDGE_MODEL,
        client=openai_client,
        max_tokens=RAGAS_JUDGE_MAX_TOKENS,
        temperature=RAGAS_JUDGE_TEMPERATURE,
    )
    run_config = RunConfig(
        timeout=RAGAS_METRIC_TIMEOUT_SEC,
        max_workers=RAGAS_MAX_WORKERS,
        max_retries=12,
    )
    embeddings = LangchainOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Format data for RAGAS (requires specific dictionary keys)
    ragas_data = {
        "question": df_results["question"].tolist(),
        "answer": df_results["generated_answer"].tolist(),
        "contexts": df_results["contexts"].tolist(),
        "ground_truth": df_results["ground_truth"].tolist()
    }
    
    ragas_dataset = Dataset.from_dict(ragas_data)
    
    # Run the math
    ragas_scores = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )
    
    # Convert RAGAS results to DataFrame and merge with our results
    df_ragas = ragas_scores.to_pandas()
    
    # Add RAGAS scores to our main dataframe
    df_results["faithfulness"] = df_ragas["faithfulness"]
    df_results["answer_relevancy"] = df_ragas["answer_relevancy"]
    df_results["answer_correctness"] = df_ragas["answer_correctness"]

    # Evaluation components report (no composite score; interpret each column separately)
    print("\n" + "=" * 50)
    print("📋 EVALUATION COMPONENTS REPORT (means over evaluated questions)")
    print("— Rule-based / retrieval —")
    print(
        f"  recall (gold in top-k): {df_results['recall_at_5'].mean() * 100:.1f}%  "
        f"(k=exp_top_k; column recall_at_5 is this flag)"
    )
    print(f"  citation_hallucination: {df_results['citation_hallucination'].mean() * 100:.1f}%")
    print(f"  gold_chunk_cited:       {df_results['gold_chunk_cited'].mean() * 100:.1f}%")
    _rec = df_results["recall_at_5"] == 1
    if _rec.any():
        print(
            f"  gold_chunk_cited | recall=1: "
            f"{df_results.loc[_rec, 'gold_chunk_cited'].mean() * 100:.1f}%"
        )
    _ch = df_results["context_hallucination"]
    _ch_valid = _ch.notna()
    if _ch_valid.any():
        rate = _ch[_ch_valid].mean() * 100
        sev = df_results.loc[_ch_valid, "context_hallucination_severity"].mean()
        _cr = df_results["claim_level_hallucination_rate"]
        _cr_valid = _cr.notna()
        claim_mean = _cr[_cr_valid].mean() * 100 if _cr_valid.any() else float("nan")
        _pos = (
            df_results["total_claims"].notna()
            & (df_results["total_claims"] > 0)
            & df_results["unsupported_claim_count"].notna()
        )
        if _pos.any():
            num_u = float(df_results.loc[_pos, "unsupported_claim_count"].sum())
            den_t = float(df_results.loc[_pos, "total_claims"].sum())
            micro = (num_u / den_t * 100) if den_t > 0 else float("nan")
        else:
            micro = float("nan")
        n_no_claims = int(
            (
                df_results["total_claims"].notna()
                & (df_results["total_claims"] == 0)
            ).sum()
        )
        print("— Context-grounded hallucination (dedicated LLM judge; passages only) —")
        print(
            f"  context_hallucination_rate:   {rate:.1f}%  "
            f"(fraction of questions with any unsupported factual content)"
        )
        print(
            f"  context_hallucination_severity (mean): {sev * 100:.1f}%  "
            f"(0=fully grounded, 100=worst; judge scale, see unsupported_claims_json in CSV)"
        )
        print(
            f"  claim_level_hallucination_rate (macro, mean over Q with claims): "
            f"{claim_mean:.1f}%  "
            f"(unsupported atomic claims / total atomic claims per answer)"
        )
        if pd.notna(micro):
            print(
                f"  claim_level_hallucination_rate (micro, pooled claims): {micro:.1f}%  "
                f"(sum unsupported / sum atomic over questions with total_claims>0)"
            )
        if n_no_claims:
            print(
                f"  note: {n_no_claims} question(s) had 0 atomic claims "
                f"(excluded from macro/micro claim rates; see atomic_claims_json)"
            )
    print("— RAGAS (LLM-as-judge on answer + contexts + ground truth) —")
    for col, label in [
        ("faithfulness", "faithfulness"),
        ("answer_relevancy", "answer_relevancy"),
        ("answer_correctness", "answer_correctness"),
    ]:
        s = df_results[col]
        n_nan = int(s.isna().sum())
        mean = s.mean(skipna=True)
        pct = (mean * 100) if pd.notna(mean) else float("nan")
        print(f"  {label}: {pct:.1f}%  (missing {n_nan}/{len(df_results)} rows)")
    print("=" * 50)

    # 5. Save audit log (per-question columns; no composite score)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_results.to_csv(out_csv, index=False)
    
    print(f"Per-question components + scores saved to: {out_csv}")

if __name__ == "__main__":
    run_evaluation_pipeline()
