import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError
from vector_store import FinancialRetriever
from experiment_config import RAGExperimentConfig, build_system_prompt, get_experiment

# --- CONFIGURATION ---
load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o"  # or gpt-3.5-turbo if you want to save money

_CITATION_RE = re.compile(r"\[Source:\s*(.*?)\]", re.DOTALL)


class AuditRAGGenerator:
    def __init__(self, experiment: RAGExperimentConfig | None = None):
        self.experiment = experiment or get_experiment("baseline")
        print(
            f"Initializing Generator & Retriever (experiment={self.experiment.experiment_id})..."
        )
        self.retriever = FinancialRetriever()

    def _format_context(self, retrieved_chunks):
        """
        Wraps chunks in explicit XML-style tags with their IDs.
        This is the secret to preventing Citation Hallucinations.
        """
        context_str = ""
        for chunk in retrieved_chunks:
            context_str += f"\n--- START CHUNK {chunk['chunk_id']} ---\n"
            context_str += f"Section: {chunk['section']}\n"
            context_str += f"{chunk['text']}\n"
            context_str += f"--- END CHUNK {chunk['chunk_id']} ---\n"
        return context_str

    @staticmethod
    def _verify_citations(llm_answer, retrieved_ids):
        """
        Post-processing check: Did the LLM cite a fake ID?
        """
        cited_ids = re.findall(r"\[Source:\s*(.*?)\]", llm_answer)
        rset = {str(r).strip() for r in retrieved_ids}

        hallucinations = []
        for cid in cited_ids:
            if cid.strip() not in rset:
                hallucinations.append(cid)

        return hallucinations

    @staticmethod
    def _is_insufficient_answer(answer: str) -> bool:
        if not isinstance(answer, str):
            return True
        t = answer.strip().lower()
        return t.startswith("insufficient information")

    @classmethod
    def _answer_has_valid_citation(cls, answer: str, retrieved_ids: list[str]) -> bool:
        rids = set(retrieved_ids)
        for m in _CITATION_RE.findall(answer):
            if m.strip() in rids:
                return True
        return False

    def answer_question(self, question, top_k: int | None = None):
        """
        The Master Generation Pipeline.
        """
        k = self.experiment.top_k if top_k is None else top_k

        print(f"Searching database for: '{question}'...")
        retrieved_chunks = self.retriever.retrieve(question, top_k=k)
        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

        context_string = self._format_context(retrieved_chunks)

        system_prompt = build_system_prompt(self.experiment.prompt_preset)

        user_prompt = f"CONTEXT:\n{context_string}\n\nQUESTION: {question}"

        print("Generating audited response...")
        delay = 1.0
        max_attempts = 12
        response = None
        for attempt in range(max_attempts):
            try:
                response = CLIENT.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                break
            except RateLimitError:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.5, 60.0)
        assert response is not None

        answer = response.choices[0].message.content

        citation_errors_pre = self._verify_citations(answer, retrieved_ids)
        is_insuff = self._is_insufficient_answer(answer)
        has_valid = self._answer_has_valid_citation(answer, retrieved_ids)
        citation_issue_pre_strict = bool(citation_errors_pre) or (
            not is_insuff and not has_valid
        )

        strict_applied = False
        if self.experiment.strict_citation_postprocess:
            if citation_issue_pre_strict:
                answer = "Insufficient Information."
                strict_applied = True

        citation_errors = self._verify_citations(answer, retrieved_ids)

        return {
            "question": question,
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieved_contexts": [c["text"] for c in retrieved_chunks],
            "citation_errors": citation_errors,
            "citation_errors_pre_strict": citation_errors_pre,
            "citation_issue_pre_strict": citation_issue_pre_strict,
            "strict_abstention_applied": strict_applied,
        }


# --- TEST BLOCK ---
if __name__ == "__main__":
    generator = AuditRAGGenerator()

    test_q = "What are the macroeconomic factors impacting Nvidia's supply chain?"

    result = generator.answer_question(test_q)

    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print(result["answer"])
    print("=" * 50)

    if len(result["citation_errors"]) > 0:
        print(f"WARNING: Citation Hallucinations Detected: {result['citation_errors']}")
    else:
        print("Citation Audit Passed. All sources are valid.")
