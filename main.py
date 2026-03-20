import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import defaultdict
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

# --- 1. REPRODUCIBILITY CONFIGURATION ---
def set_seed(seed=42):
    """Set global seeds for deterministic behavior across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# --- 2. MAIN EXECUTION LOGIC ---
def main():
    set_seed(42)
    
    # Device setup: Optimized for Apple Silicon (M4) or fallback to CPU

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print("="*40)
    print(f"✅ Imports completed successfully.")
    print(f"💻 Active Device: {device.upper()}")
    print("="*40)

    # --- PHASE A: DATA ACQUISITION AND PREPROCESSING ---
    print("\n--- Initializing data download ---")
    corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")

    print(f"Downloaded: {len(corpus)} documents, {len(queries)} queries, {len(qrels)} relations.")

    # Convert datasets to lookup dictionaries for evaluation efficiency
    corpus_dict = {doc['_id']: doc['text'] for doc in corpus}
    queries_dict = {q['_id']: q['text'] for q in queries}

    # Map query IDs to sets of relevant document IDs (Ground Truth)
    relevant_docs = defaultdict(set)
    for rel in qrels:
        relevant_docs[rel['query-id']].add(rel['corpus-id'])

    print("\nEvaluation Data Prepared:")
    print(f"- Documents in corpus: {len(corpus_dict)}")
    print(f"- Test queries: {len(queries_dict)}")

    # --- PHASE B: BASELINE PERFORMANCE EVALUATION ---
    print(f"\nLoading base model onto {device}...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    # Initialize the Information Retrieval evaluator
    ir_evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries_dict,
        corpus=corpus_dict,
        relevant_docs=relevant_docs,
        name='baseline',
        show_progress_bar=True
    )

    results = ir_evaluator(model)

    print("\n" + "="*30)
    print("Baseline Metrics")
    print("="*30)
    print(f"Recall@10: {results['baseline_cosine_recall@10']:.4f}")
    print(f"NDCG@10:   {results['baseline_cosine_ndcg@10']:.4f}")
    print(f"MRR@10:    {results['baseline_cosine_mrr@10']:.4f}")

    # --- PHASE C: TRAINING DATASET PREPARATION ---
    print("\n--- Preparing fine-tuning dataset ---")
    train_qrels = load_dataset("BeIR/nfcorpus-qrels", split="train")

    # Aggregate document IDs by query for controlled sampling
    train_query2docs = defaultdict(list)
    for rel in train_qrels:
        train_query2docs[rel['query-id']].append(rel['corpus-id'])

    train_examples = []
    max_pos_per_query = 7  # Limit positive samples per query to prevent bias

    for q_id, doc_ids in train_query2docs.items():
        query_text = queries_dict.get(q_id)
        if not query_text:
            continue

        for d_id in doc_ids[:max_pos_per_query]:
            doc_text = corpus_dict.get(d_id)
            if not doc_text:
                continue
            # Format as InputExample for SentenceTransformers
            train_examples.append(InputExample(texts=[query_text, doc_text]))
            
    print(f"Generated {len(train_examples)} training examples.")

    # --- PHASE D: MODEL FINE-TUNING ---
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    
    # Using MultipleNegativesRankingLoss for contrastive learning
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Hyperparameter configuration
    num_epochs = 4
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) # 10% of total steps
    model.max_seq_length = 256 

    print("\n--- Starting Fine-Tuning Process ---")
    print(f"Epochs: {num_epochs} | Max Seq Length: {model.max_seq_length} | Warmup Steps: {warmup_steps}")
    
    use_amp = True if device == "cuda" else False  # AMP stable on CUDA, unstable on MPS

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': 1.5e-5},
        output_path='tuned_model_nfcorpus_v10',
        save_best_model=True,
        show_progress_bar=True,
        use_amp=use_amp,          # True on CUDA, False on MPS/CPU
        evaluation_steps=50       # Frequent evaluation to capture best checkpoint
    )
    print("\n✅ Training completed. Best model checkpoint saved to 'tuned_model_nfcorpus_v10'.")

# --- 3. ENTRY POINT ---
if __name__ == "__main__":
    main()