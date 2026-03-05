import os
import json
import numpy as np
import pickle
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
load_dotenv()
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.jsonl")
CHROMA_PATH = "outputs/chroma_db"
BM25_PATH = "outputs/bm25_model.pkl"

class FinancialRetriever:
    def __init__(self):
        """
        Initializes the Hybrid Retriever (ChromaDB + BM25 + Reranker).
        """
        print("🔧 Initializing Financial Retriever...")
        
        # 1. Setup ChromaDB (Vector Store)
        # We use OpenAI Embeddings (High quality, cheap)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name="finance_docs",
            embedding_function=self.openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 2. Setup Reranker (Cross-Encoder)
        # Uses a local HuggingFace model (Free & State-of-the-Art)
        print("   -> Loading Reranker Model (BAAI/bge-reranker-base)...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        
        # 3. Setup BM25 (Keyword Store)
        # We load this lazily or build it if missing
        self.bm25 = None
        self.chunks_map = {} # Maps chunk_id -> full_chunk_data
        self.corpus_tokens = []
        
        # Load data into memory (for BM25 and ID lookups)
        self._load_corpus()

    def _load_corpus(self):
        """Loads corpus.jsonl into memory and builds/loads BM25."""
        if not os.path.exists(CORPUS_PATH):
            raise FileNotFoundError(f"Missing {CORPUS_PATH}. Run etl_pipeline.py first.")

        print("   -> Loading Corpus...")
        documents = []
        ids = []
        
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks_map[chunk['chunk_id']] = chunk
                documents.append(chunk['text'])
                ids.append(chunk['chunk_id'])
                
        # Build BM25
        print(f"   -> Building BM25 Index for {len(documents)} documents...")
        tokenized_corpus = [doc.lower().split() for doc in documents] # Simple tokenizer
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_ids = ids # Keep track of which index maps to which ID
        print("✅ Retriever Ready.")

    def build_index(self):
        """
        Pushes data to ChromaDB. 
        Only run this once (or when data changes).
        """
        print("🏗️ Building Vector Index...")
        
        ids = []
        documents = []
        metadatas = []
        
        # Check if already populated to save money
        if self.collection.count() > 0:
            print(f"   -> Collection already has {self.collection.count()} docs. Skipping ingestion.")
            return

        for chunk_id, chunk in self.chunks_map.items():
            ids.append(chunk_id)
            documents.append(chunk['text'])
            metadatas.append({
                "source": chunk['source'],
                "section": chunk['section']
            })
            
        # Add in batches (Chroma likes batches)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            print(f"   -> Indexed {min(i+batch_size, len(ids))}/{len(ids)}")
            
        print("✅ Indexing Complete.")

    def retrieve(self, query, top_k=5):
        """
        The Master Retrieval Function:
        1. Dense Search (Chroma) -> Top 10
        2. Sparse Search (BM25) -> Top 10
        3. Merge & Deduplicate
        4. Rerank -> Top 5
        """
        # A. Vector Search (Semantic)
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=10
        )
        vector_ids = vector_results['ids'][0]

        # B. Keyword Search (Lexical)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top 10 indices
        top_n_indices = np.argsort(bm25_scores)[::-1][:10]
        bm25_ids = [self.doc_ids[i] for i in top_n_indices]

        # C. Ensemble (Merge)
        combined_ids = list(set(vector_ids + bm25_ids)) # Remove duplicates
        
        # D. Reranking (The Judge)
        # Prepare pairs for the Cross-Encoder: [[Query, Text1], [Query, Text2]...]
        pairs = []
        for cid in combined_ids:
            chunk_text = self.chunks_map[cid]['text']
            pairs.append([query, chunk_text])
            
        scores = self.reranker.predict(pairs)
        
        # Sort by score (High to Low)
        sorted_indices = np.argsort(scores)[::-1]
        
        # Format the Final Output
        final_results = []
        for idx in sorted_indices[:top_k]:
            cid = combined_ids[idx]
            final_results.append(self.chunks_map[cid])
            
        return final_results

# Simple Test Block
if __name__ == "__main__":
    retriever = FinancialRetriever()
    
    # Run Indexing (Only does work if DB is empty)
    retriever.build_index()
    
    # Test Query
    query = "What was the revenue for 2024?"
    results = retriever.retrieve(query)
    
    print(f"\n🔍 Query: {query}")
    print(f"🏆 Top Result ID: {results[0]['chunk_id']}")
    print(f"📄 Section: {results[0]['section']}")
    print(f"📝 Text Snippet: {results[0]['text'][:200]}...")