import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import freeze_support
import time
import os

# --- BPE Tokenizer Imports ---
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- HYPERPARAMETER SEARCH SPACE ---
param_grid = {
    'PRETRAIN_EPOCHS': [5, 10],
    'PRETRAIN_LR': [0.001, 0.005],
    'WINDOW_SIZE': [2, 4]  # Word-level window
}

# --- FIXED HYPERPARAMETERS ---
BPE_VOCAB_SIZE = 8000 # The size of our subword vocabulary
EMBEDDING_DIM = 128
MAIN_LR = 0.001
BATCH_SIZE = 512 # Reduced batch size as sequences can be longer
MAIN_EPOCHS = 10

# --- 0. DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (Unchanged) ---
# This model is flexible and already handles variable input lengths correctly.
class AttentionNgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(AttentionNgram, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # Use padding_idx
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
        # We sum over the sequence dimension to get a single context vector
        context_vec = torch.matmul(attn_weights, v).sum(dim=1)
        out = F.relu(self.fc1(context_vec))
        out = self.dropout(out); out = self.fc2(out)
        return out

# --- HELPER FUNCTIONS ---

def create_dummy_corpus(filename="input.txt"):
    """Creates a sample input.txt if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"'{filename}' not found. Creating a dummy file for demonstration.")
        text = """
        Language models are a type of model that is trained to predict the next word in a sequence of words.
        This is known as language modeling. The most successful language models are based on the Transformer architecture.
        Pre-training language models on a large corpus of text has been shown to be a very effective technique.
        This process, often called self-supervised learning, allows the model to learn general-purpose representations of language.
        These representations can then be fine-tuned on a smaller, task-specific dataset.
        Byte-Pair Encoding, or BPE, is a tokenization technique that helps models handle rare words by breaking them into smaller, more common subword units.
        This script demonstrates priming the embedding space using a BPE tokenizer and a CBOW-style pre-training objective.
        The goal is to find the best pre-training parameters to warm up the embeddings for a downstream task.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(" ".join(text.lower().split()))

def train_bpe_tokenizer(filepath, vocab_size):
    """Trains a BPE tokenizer from a text file."""
    print("\n1a. Training BPE Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    
    tokenizer.train(files=[filepath], trainer=trainer)
    # Important: Set padding token and enable it
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    
    print(f"   Tokenizer training complete. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

class BpeTextDataset(Dataset):
    """Custom PyTorch Dataset for our BPE-based tasks."""
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        context_ids, target_id = self.data_pairs[idx]
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

def create_collate_fn(pad_token_id):
    """Creates a collate function to handle padding for variable length sequences."""
    def collate_fn(batch):
        contexts, targets = zip(*batch)
        # Pad contexts to the max length in the batch
        padded_contexts = torch.nn.utils.rnn.pad_sequence(contexts, batch_first=True, padding_value=pad_token_id)
        # Stack targets into a single tensor
        targets = torch.stack(targets)
        return padded_contexts, targets
    return collate_fn

def generate_bpe_data_pairs(original_words, tokenizer, context_size, is_cbow=False):
    """
    Generates context-target pairs using BPE, based on word-level windows.
    - If is_cbow=False (N-gram task): context is preceding words, target is the next word.
    - If is_cbow=True (CBOW task): context is surrounding words, target is the center word.
    """
    data_pairs = []
    window = context_size if is_cbow else 0
    
    for i in range(context_size, len(original_words) - window):
        if is_cbow:
            # CBOW: context is surrounding words, target is center word
            context_words = original_words[i-context_size:i] + original_words[i+1:i+context_size+1]
            target_word = original_words[i]
        else:
            # N-gram: context is preceding words, target is next word
            context_words = original_words[i-context_size:i]
            target_word = original_words[i]

        context_sentence = " ".join(context_words)
        context_ids = tokenizer.encode(context_sentence).ids
        target_ids = tokenizer.encode(target_word).ids

        # We need both context and target to be valid (non-empty)
        if not context_ids or not target_ids:
            continue
        
        # The model will predict the *first* subword token of the target word
        first_target_id = target_ids[0]
        data_pairs.append((context_ids, first_target_id))
        
    return data_pairs

def run_cbow_pretraining(vocab_size, cbow_dataloader, pretrain_epochs, pretrain_lr, run_id_str):
    """Trains a CBOW model and returns the learned embeddings."""
    # A simple model: an embedding layer (with padding) and a linear layer
    cbow_model = nn.Sequential(
        nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0),
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
            # Average the embeddings, ignoring padding by checking for non-zero IDs
            mask = (context != 0).unsqueeze(-1).expand_as(embedded_context)
            sum_embeds = (embedded_context * mask).sum(1)
            non_pad_elements = mask.sum(1)
            non_pad_elements[non_pad_elements == 0] = 1 # Avoid division by zero
            avg_context = sum_embeds / non_pad_elements
            
            logits = cbow_model[1](avg_context)
            loss = loss_fn(logits, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    return cbow_model[0].weight.data.clone()

def run_main_task_training(vocab_size, main_task_dataloader, initial_embeddings=None, run_id_str=""):
    """Trains the main AttentionNgram model and returns the final average loss."""
    model = AttentionNgram(vocab_size, EMBEDDING_DIM).to(device)
    
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
    
    # --- 1. PREPARE CORPUS AND TOKENIZER ---
    corpus_file = "input.txt"
    create_dummy_corpus(corpus_file)
    tokenizer = train_bpe_tokenizer(corpus_file, BPE_VOCAB_SIZE)
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    collate_fn = create_collate_fn(pad_token_id)

    with open(corpus_file, 'r', encoding='utf-8') as f:
        original_words = f.read().split()

    # --- 2. PREPARE MAIN TASK DATALOADER (ONCE) ---
    print("\n1b. Preparing Main Task Data (N-gram prediction)...")
    # Context size for main task: use 3 preceding words
    main_task_pairs = generate_bpe_data_pairs(original_words, tokenizer, context_size=3, is_cbow=False)
    main_task_dataset = BpeTextDataset(main_task_pairs)
    main_task_dataloader = DataLoader(main_task_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"   Generated {len(main_task_pairs)} examples for the main task.")

    # --- 3. RUN BASELINE (ONCE) ---
    print("\n--- (A) Training Main Task from Scratch (Baseline) ---")
    baseline_loss = run_main_task_training(vocab_size, main_task_dataloader)
    print(f"   Baseline (Random Init) Final Avg Loss: {baseline_loss:.4f}")

    # --- 4. HYPERPARAMETER GRID SEARCH ---
    print("\n--- (B) Hyperparameter Search for CBOW Pre-training ---")
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results, best_loss, best_params = [], float('inf'), None

    for i, params in enumerate(param_combinations):
        run_id_str = f"[Run {i+1}/{len(param_combinations)}]"
        print(f"\n{run_id_str} Testing params: {params}")
        
        # a) Create CBOW data and dataloader for the current window size
        cbow_pairs = generate_bpe_data_pairs(original_words, tokenizer, context_size=params['WINDOW_SIZE'], is_cbow=True)
        print(f"   Generated {len(cbow_pairs)} CBOW pairs for pre-training.")
        cbow_dataset = BpeTextDataset(cbow_pairs)
        cbow_dataloader = DataLoader(cbow_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

        # b) Run CBOW pre-training
        warmed_up_embeddings = run_cbow_pretraining(vocab_size, cbow_dataloader, params['PRETRAIN_EPOCHS'], params['PRETRAIN_LR'], run_id_str)
        
        # c) Run main task with warmed-up embeddings
        final_loss = run_main_task_training(vocab_size, main_task_dataloader, initial_embeddings=warmed_up_embeddings, run_id_str=run_id_str)
        print(f"   {run_id_str} Final Warmed-up Avg Loss: {final_loss:.4f}")
        
        # d) Store results
        current_run_results = params.copy()
        current_run_results['final_loss'] = final_loss
        results.append(current_run_results)
        
        if final_loss < best_loss:
            best_loss, best_params = final_loss, params

    # --- 5. FINAL RESULTS ---
    print("\n\n--- HYPERPARAMETER SEARCH COMPLETE ---")
    print(f"Total time elapsed: {(time.time() - start_time)/60:.2f} minutes")
    print("-" * 50)
    print(f"Baseline (Random Init) Final Avg Loss: {baseline_loss:.4f}")
    print("-" * 50)

    if best_params:
        print(f"🏆 Best Warmed-up Final Avg Loss: {best_loss:.4f}")
        print(f"   Achieved with parameters: {best_params}")
        boost = (baseline_loss - best_loss) / baseline_loss * 100
        print(f"   Performance Boost vs. Baseline: {boost:.2f}%")
    else:
        print("No successful pre-training runs were completed.")

    print("\n--- Full Results Summary ---")
    header = " | ".join([f"{k:<15}" for k in param_grid.keys()] + ["Final Loss"])
    print(header)
    print("-" * len(header))
    for res in sorted(results, key=lambda x: x['final_loss']):
        row_str = " | ".join([f"{str(v):<15}" for k, v in res.items() if k != 'final_loss'] + [f"{res['final_loss']:.4f}"])
        if res == best_params:
             row_str += "  <-- BEST"
        print(row_str)
    print("-" * 50)


if __name__ == '__main__':
    freeze_support()
    main()