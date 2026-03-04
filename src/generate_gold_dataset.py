import os
import json
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- CONFIGURATION ---
load_dotenv()
CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths
INPUT_CORPUS = "data/processed/corpus.jsonl"
OUTPUT_CSV = "data/synthetic/gold_dataset.csv"

# Hyperparameters
SAMPLE_SIZE = 50          # How many chunks to attempt
MIN_CHUNK_LENGTH = 300    # Skip tiny chunks (headers/footers)
MODEL_NAME = "gpt-4o"     # The Teacher (High Intelligence)

def load_chunks(filepath):
    """Load and filter valid chunks from JSONL."""
    chunks = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            # Filter: Don't generate questions from empty lines or short footers
            if len(chunk['text']) > MIN_CHUNK_LENGTH:
                chunks.append(chunk)
    print(f"✅ Loaded {len(chunks)} valid chunks from corpus.")
    return chunks

def generate_question(chunk):
    """
    THE TEACHER: Generates a specific financial question based on the text.
    """
    prompt = f"""
    You are a Senior Financial Analyst constructing a test for a junior analyst.
    
    CONTEXT (Source: {chunk['source']} | Section: {chunk['section']}):
    {chunk['text']}
    
    TASK:
    1. Generate a specific, difficult question that can be answered ONLY using the context above.
    2. Provide the correct answer based on the context.
    3. The question should involve specific numbers, dates, or distinct risk factors if present.
    
    OUTPUT JSON FORMAT:
    {{
        "question": "...",
        "answer": "..."
    }}
    """
    
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": "You are a strict financial auditor."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7 # Slight creativity for diverse questions
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        return None

def critique_question(question, answer, chunk_text):
    """
    THE CRITIQUE AGENT: Filters out bad questions.
    Returns True if it passes, False if it fails.
    """
    prompt = f"""
    You are a RAG Evaluation Auditor. Grade this Question-Answer pair based on the Context.
    
    CONTEXT:
    {chunk_text}
    
    PROPOSED QUESTION: {question}
    PROPOSED ANSWER: {answer}
    
    CRITERIA (Score 1-5):
    1. Groundedness: Is the answer fully supported by the context? (5 = Fully supported)
    2. Standalone: Does the question make sense without seeing the document? (e.g. "What is the revenue?" is bad. "What was Nvidia's 2024 revenue?" is good).
    3. Difficulty: Is it non-trivial?
    
    OUTPUT JSON FORMAT:
    {{
        "groundedness_score": <int>,
        "standalone_score": <int>,
        "total_score": <int>,
        "pass": <bool> (True only if groundedness >= 4 AND standalone >= 4)
    }}
    """
    
    try:
        response = CLIENT.chat.completions.create(
            model=MODEL_NAME, # Use strict model for grading
            messages=[{"role": "system", "content": "You are a critical auditor."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0 # Strict, deterministic grading
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"❌ Critique Error: {e}")
        return None

def main():
    # 1. Load Data
    all_chunks = load_chunks(INPUT_CORPUS)
    
    # 2. Sample
    if len(all_chunks) < SAMPLE_SIZE:
        selected_chunks = all_chunks
    else:
        selected_chunks = random.sample(all_chunks, SAMPLE_SIZE)
        
    print(f"🎲 Selected {len(selected_chunks)} chunks for generation...")
    
    gold_data = []
    
    # 3. Iterate (Generate -> Critique -> Save)
    for chunk in tqdm(selected_chunks, desc="Generating QA Pairs"):
        # A. Generate
        qa_pair = generate_question(chunk)
        if not qa_pair: continue
        
        # B. Critique
        critique = critique_question(qa_pair['question'], qa_pair['answer'], chunk['text'])
        
        # C. Filter
        if critique and critique['pass']:
            gold_data.append({
                "question": qa_pair['question'],
                "ground_truth_answer": qa_pair['answer'],
                "gold_chunk_id": chunk['chunk_id'], # CRITICAL FOR RECALL@K
                "source_doc": chunk['source'],
                "section": chunk['section'],
                "groundedness_score": critique['groundedness_score']
            })
        else:
            # Optional: Print rejected questions to see why they failed
            # print(f"Rejected: {qa_pair['question']} (Score: {critique})")
            pass

    # 4. Save to CSV
    df = pd.DataFrame(gold_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Generation Complete!")
    print(f"Total Attempted: {len(selected_chunks)}")
    print(f"Total Passed Critique: {len(df)}")
    print(f"💾 Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()