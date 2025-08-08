import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from collections import Counter
import math
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import freeze_support
import time

# --- HYPERPARAMETER SEARCH SPACE ---
# We will search over these pre-training parameters
param_grid = {
    'PRETRAIN_EPOCHS': [5, 10],
    'PRETRAIN_LR': [0.005, 0.01, 0.05],
    'WINDOW_SIZE': [2, 3, 5, 7, 10, 20, 64]
}

# --- FIXED HYPERPARAMETERS ---
# Main Task Hyperparameters (kept constant for a fair comparison)
CONTEXT_SIZE = 3
EMBEDDING_DIM = 128
MAIN_LR = 0.001
BATCH_SIZE = 1024
MAIN_EPOCHS = 10 # Kept relatively low to make the search faster

# --- 0. DEVICE SETUP (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Unchanged) ---
class AttentionNgram(nn.Module):
    """Our main task model."""
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(AttentionNgram, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        import torch.nn.functional as F
        embeds = self.embeddings(inputs)
        q = self.q_proj(embeds); k = self.k_proj(embeds); v = self.v_proj(embeds)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context_vec = torch.matmul(attn_weights, v).sum(dim=1)
        out = F.relu(self.fc1(context_vec))
        out = self.dropout(out); out = self.fc2(out)
        return out

# --- REFACTORED FUNCTIONS ---

def prepare_data():
    """Reads the corpus, creates vocab and main task dataloader."""
    print("\n1. Preparing Data...")
    try:
        with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
    except FileNotFoundError:
        print("Error: `input.txt` not found. Please create this file.")
        exit()
    
    tokens = text.lower().split()
    vocab = sorted(list(set(tokens)))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    vocab_size = len(vocab)
    corpus_ix = [word_to_ix.get(w) for w in tokens if w in word_to_ix]
    
    # Data for Main Task (N-grams)
    ngrams = [(corpus_ix[i:i + CONTEXT_SIZE], corpus_ix[i + CONTEXT_SIZE]) for i in range(len(corpus_ix) - CONTEXT_SIZE)]
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Generated {len(ngrams)} n-grams for main task.")
    main_task_contexts = torch.tensor([ng[0] for ng in ngrams])
    main_task_targets = torch.tensor([ng[1] for ng in ngrams])
    main_task_dataset = TensorDataset(main_task_contexts, main_task_targets)
    main_task_dataloader = DataLoader(main_task_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return vocab_size, corpus_ix, main_task_dataloader

def prepare_cbow_dataloader(corpus_ix, window_size, batch_size):
    """Prepares the CBOW dataloader for a given window size."""
    cbow_data = []
    for i in range(window_size, len(corpus_ix) - window_size):
        context = corpus_ix[i-window_size:i] + corpus_ix[i+1:i+window_size+1]
        target = corpus_ix[i]
        cbow_data.append((context, target))
    
    if not cbow_data:
        return None # Handle cases where window size is too large for the corpus

    cbow_contexts = torch.tensor([d[0] for d in cbow_data])
    cbow_targets = torch.tensor([d[1] for d in cbow_data])
    cbow_dataset = TensorDataset(cbow_contexts, cbow_targets)
    return DataLoader(cbow_dataset, batch_size=batch_size, shuffle=True)

def run_cbow_pretraining(vocab_size, cbow_dataloader, pretrain_epochs, pretrain_lr, run_id_str):
    """Trains a CBOW model and returns the learned embeddings."""
    cbow_model = nn.Sequential(
        nn.Embedding(vocab_size, EMBEDDING_DIM),
        nn.Linear(EMBEDDING_DIM, vocab_size)
    ).to(device)
    optimizer = optim.Adam(cbow_model.parameters(), lr=pretrain_lr)
    loss_fn = nn.CrossEntropyLoss()

    cbow_model.train()
    for epoch in range(pretrain_epochs):
        pbar = tqdm(cbow_dataloader, desc=f"   {run_id_str} CBOW Epoch {epoch+1}/{pretrain_epochs}", leave=False)
        for context, target in pbar:
            context, target = context.to(device), target.to(device)
            embedded_context = cbow_model[0](context)
            avg_context = torch.mean(embedded_context, dim=1)
            logits = cbow_model[1](avg_context)
            loss = loss_fn(logits, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return cbow_model[0].weight.data.clone()

def run_main_task_training(vocab_size, main_task_dataloader, initial_embeddings=None, run_id_str=""):
    """Trains the main AttentionNgram model and returns the final average loss."""
    model = AttentionNgram(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE).to(device)
    
    # Initialize embeddings: either warmed-up or random
    if initial_embeddings is not None:
        model.embeddings.weight.data.copy_(initial_embeddings)
        desc_prefix = f"   {run_id_str} Warmed-up"
    else:
        desc_prefix = "   Baseline"

    optimizer = optim.Adam(model.parameters(), lr=MAIN_LR)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(MAIN_EPOCHS):
        total_loss = 0
        pbar = tqdm(main_task_dataloader, desc=f"{desc_prefix} Epoch {epoch+1}/{MAIN_EPOCHS}", leave=False)
        for context, target in pbar:
            log_probs = model(context.to(device))
            loss = loss_fn(log_probs, target.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
    return total_loss / len(main_task_dataloader)

# --- MAIN EXECUTION BLOCK ---
def main():
    start_time = time.time()
    print(f"--- Using device: {device} ---")
    
    # --- 1. SINGLE DATA PREP ---
    vocab_size, corpus_ix, main_task_dataloader = prepare_data()

    # --- 2. RUN BASELINE (ONCE) ---
    print("\n--- (A) Training Main Task from Scratch (Baseline) ---")
    baseline_loss = run_main_task_training(vocab_size, main_task_dataloader)
    print(f"   Baseline (Random Init) Final Avg Loss: {baseline_loss:.4f}")

    # --- 3. HYPERPARAMETER GRID SEARCH ---
    print("\n--- (B) Hyperparameter Search for CBOW Pre-training ---")
    
    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    best_loss = float('inf')
    best_params = None

    for i, params in enumerate(param_combinations):
        run_id = i + 1
        pretrain_epochs = params['PRETRAIN_EPOCHS']
        pretrain_lr = params['PRETRAIN_LR']
        window_size = params['WINDOW_SIZE']
        run_id_str = f"[Run {run_id}/{len(param_combinations)}]"
        
        print(f"\n{run_id_str} Testing params: {params}")

        # a) Create CBOW data for the current window size
        cbow_dataloader = prepare_cbow_dataloader(corpus_ix, window_size, BATCH_SIZE)
        if cbow_dataloader is None:
            print(f"   Skipping: Window size {window_size} is too large for the corpus.")
            continue
        print(f"   Generated {len(cbow_dataloader.dataset)} CBOW pairs for pre-training.")

        # b) Run CBOW pre-training
        warmed_up_embeddings = run_cbow_pretraining(vocab_size, cbow_dataloader, pretrain_epochs, pretrain_lr, run_id_str)
        
        # c) Run main task with warmed-up embeddings
        final_loss = run_main_task_training(vocab_size, main_task_dataloader, initial_embeddings=warmed_up_embeddings, run_id_str=run_id_str)
        
        print(f"   {run_id_str} Final Warmed-up Avg Loss: {final_loss:.4f}")
        
        # d) Store results
        current_run_results = params.copy()
        current_run_results['final_loss'] = final_loss
        results.append(current_run_results)
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = params

    # --- 4. FINAL RESULTS ---
    print("\n\n--- HYPERPARAMETER SEARCH COMPLETE ---")
    print(f"Total time elapsed: {(time.time() - start_time)/60:.2f} minutes")
    print("-" * 40)
    print(f"Baseline (Random Init) Final Avg Loss: {baseline_loss:.4f}")
    print("-" * 40)

    if best_params:
        print(f"🏆 Best Warmed-up Final Avg Loss: {best_loss:.4f}")
        print(f"   Achieved with parameters: {best_params}")
        boost = (baseline_loss - best_loss) / baseline_loss * 100
        print(f"   Performance Boost vs. Baseline: {boost:.2f}%")
    else:
        print("No successful pre-training runs were completed.")

    # Print a summary table of all results
    print("\n--- Full Results Summary ---")
    # Header
    header = " | ".join([f"{k:<15}" for k in param_grid.keys()] + ["Final Loss"])
    print(header)
    print("-" * len(header))
    # Rows
    sorted_results = sorted(results, key=lambda x: x['final_loss'])
    for res in sorted_results:
        row_str = " | ".join([f"{str(v):<15}" for v in res.values()])
        if res == best_params:
             row_str += "  <-- BEST"
        print(row_str)
    print("-" * 40)


if __name__ == '__main__':
    freeze_support() # Important for multiprocessing/windows
    main()