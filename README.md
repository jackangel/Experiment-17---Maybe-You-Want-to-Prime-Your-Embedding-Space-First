# Maybe You Want to Prime Your Embedding Space First?

This repository contains a small-scale research experiment designed to explore a simple question: **Does pre-training an embedding layer with a CBOW-style objective improve performance on a downstream NLP task?**

Instead of taking pre-trained embeddings like GloVe or Word2Vec off the shelf, this project investigates a "warm-up" or "priming" strategy where we learn embeddings *from the target corpus itself* using a simple, self-supervised task before training the main, more complex model.

The experiment is conducted under two different tokenization strategies:
1.  **Simple Word-Level Tokenization:** Each unique word in the vocabulary gets a unique token.
2.  **Byte-Pair Encoding (BPE) Subword Tokenization:** Words are broken down into smaller, more common subword units.

## The Experiment

The core methodology for both scenarios is as follows:

### 1. The Downstream Task
The main goal is to train an `AttentionNgram` model. This model uses a self-attention mechanism over a context of preceding words (or subwords) to predict the next word (or its first subword token). This serves as our benchmark task.

### 2. The Pre-Training Strategy
We use a **Continuous Bag-of-Words (CBOW)** approach for pre-training. A simple neural network is trained to predict a center word given its surrounding context words. The sole purpose of this step is to generate meaningful embeddings. The trained CBOW model itself is then discarded.

### 3. The Methodology
For each tokenization strategy, we perform a comparison:
*   **Baseline:** Train the `AttentionNgram` model from scratch with randomly initialized embeddings.
*   **Warmed-up:**
    1.  First, train a simple CBOW model on the corpus for a few epochs.
    2.  Extract the learned embedding layer's weights.
    3.  Initialize the `AttentionNgram` model using these "warmed-up" embeddings.
    4.  Train the `AttentionNgram` model on the same downstream task.
*   **Hyperparameter Search:** We run a grid search over the pre-training parameters (learning rate, epochs, and context window size) to find the combination that gives the warmed-up model the biggest advantage.

The final average loss on the downstream task is used to measure success. A lower loss is better.

---

## How to Run

### Prerequisites
*   Python 3.x
*   PyTorch
*   Tqdm
*   NumPy
*   `tokenizers` library (from Hugging Face)

You can install the dependencies with:
```bash
pip install torch numpy tqdm tokenizers
```

### Setup
1.  Clone this repository.
2.  Create a text file named `input.txt` in the root directory and populate it with your training corpus. The scripts are case-insensitive and expect whitespace-separated text.
    > **Note:** The `cbow_warmup_researcher_bpe.py` script will create a small dummy `input.txt` if one is not found.

### Execution
To run the experiments, execute the python scripts from your terminal.

**For the word-level experiment:**
```bash
python cbow_warmup_researcher.py
```

**For the BPE subword-level experiment:**
```bash
python cbow_warmup_researcher_bpe.py
```

The scripts will run the baseline, then iterate through the hyperparameter grid, and finally print a summary of the results.

---

## Results & Analysis

Here are the results from running the experiments on a sample corpus.

### Scenario 1: Word-Level Tokenization (`cbow_warmup_researcher.py`)

The baseline model with random embeddings achieved a final average loss of **4.1982**. The best-performing warmed-up model significantly outperformed this.

> **🏆 Best Warmed-up Performance**
> *   **Final Avg Loss:** **3.0984**
> *   **Performance Boost vs. Baseline:** **26.20%**
> *   **Best Parameters:** `{'PRETRAIN_EPOCHS': 10, 'PRETRAIN_LR': 0.01, 'WINDOW_SIZE': 3}`

#### Full Results Summary

| PRETRAIN_EPOCHS | PRETRAIN_LR     | WINDOW_SIZE     | Final Loss |
|:----------------|:----------------|:----------------|:-----------|
| 10              | 0.01            | 3               | **3.0984**     |
| 5               | 0.05            | 3               | 3.1872     |
| 10              | 0.05            | 3               | 3.1933     |
| 5               | 0.01            | 3               | 3.2833     |
| ...             | ...             | ...             | ...        |
| 10              | 0.01            | 64              | 4.1333     |
| 10              | 0.05            | 20              | 5.0919     |
| 5               | 0.05            | 64              | 5.2038     |
| 10              | 0.05            | 64              | 5.4134     |

#### Analysis
With simple word-level tokenization, priming the embedding space shows a **clear and significant benefit**. The 26% improvement suggests that helping the model start with a better semantic representation is highly effective.

However, the results also show that this is not a magic bullet. The choice of pre-training hyperparameters is critical. Notice how a very large `WINDOW_SIZE` (e.g., 20 or 64) combined with a high `PRETRAIN_LR` (0.05) resulted in performance *worse than the baseline*. This indicates that "bad" pre-training can be more harmful than no pre-training at all.

---

### Scenario 2: BPE Subword Tokenization (`cbow_warmup_researcher_bpe.py`)

The baseline BPE model achieved a final average loss of **5.0468**. The improvement from the warmed-up model was more modest.

> **🏆 Best Warmed-up Performance**
> *   **Final Avg Loss:** **4.7315**
> *   **Performance Boost vs. Baseline:** **6.25%**
> *   **Best Parameters:** `{'PRETRAIN_EPOCHS': 10, 'PRETRAIN_LR': 0.005, 'WINDOW_SIZE': 4}`

#### Full Results Summary

| PRETRAIN_EPOCHS | PRETRAIN_LR     | WINDOW_SIZE     | Final Loss |
|:----------------|:----------------|:----------------|:-----------|
| 10              | 0.005           | 4               | **4.7315**     |
| 10              | 0.005           | 2               | 4.7356     |
| 5               | 0.005           | 2               | 4.7873     |
| 5               | 0.005           | 4               | 4.7899     |
| 10              | 0.001           | 2               | 4.8997     |
| 10              | 0.001           | 4               | 4.9018     |
| 5               | 0.001           | 2               | 4.9476     |
| 5               | 0.001           | 4               | 4.9485     |


#### Analysis
With a more sophisticated BPE tokenizer, the benefit of priming is **still present but much smaller**. A 6.25% boost is welcome, but it's not the game-changer seen in the word-level experiment.

This might suggest a few things:
1.  BPE tokenization, by its nature, already creates a more structured vocabulary where related subwords share representations. This might give the random-initialization baseline a better starting point, reducing the relative gain from pre-training.
2.  The downstream task (predicting the *first* subword of the next word) might be slightly different or easier than predicting a full word, changing the dynamics.
3.  The hyperparameter search space was smaller; a wider search might uncover parameters that yield a larger boost.





### Scenario 3: Word-Level Tokenization (`skipgram_warmup_researcher.py`)

**Baseline (Random Init) Final Avg Loss:** 4.1602

---

🏆 **Best Warmed-up Final Avg Loss:** 2.9610
> Achieved with parameters: `{'PRETRAIN_EPOCHS': 10, 'PRETRAIN_LR': 0.005, 'WINDOW_SIZE': 5}`
> **Performance Boost vs. Baseline:** 28.82%

### Full Results Summary

| PRETRAIN_EPOCHS | PRETRAIN_LR | WINDOW_SIZE | Final Loss |
|:---------------:|:-----------:|:-----------:|:---------------------|
| 10 | 0.005 | 5 | 2.9610409921643326 **&lt;-- BEST** |
| 5 | 0.005 | 5 | 3.0207935737481404 |
| 10 | 0.005 | 7 | 3.0446952792596327 |
| 5 | 0.005 | 7 | 3.065512686487871 |
| 5 | 0.01 | 5 | 3.074102272226381 |
| 10 | 0.01 | 5 | 3.0804032612684704 |
| 10 | 0.01 | 3 | 3.1304391226342623 |
| 10 | 0.005 | 10 | 3.1551501436778078 |
| 5 | 0.005 | 10 | 3.155809900945661 |
| 5 | 0.01 | 7 | 3.185602737904293 |
| 5 | 0.01 | 3 | 3.206088099612324 |
| 10 | 0.01 | 7 | 3.2085226995878653 |
| 10 | 0.005 | 3 | 3.2392935693525815 |
| 10 | 0.01 | 10 | 3.3207756586340125 |
| 5 | 0.01 | 10 | 3.325335318312638 |
| 5 | 0.005 | 3 | 3.3919142531790114 |
| 10 | 0.01 | 2 | 3.4686652797207476 |
| 5 | 0.01 | 2 | 3.5907795851611115 |
| 10 | 0.005 | 2 | 3.6408449901843314 |
| 5 | 0.005 | 2 | 3.773753499216775 |
| 10 | 0.05 | 3 | 4.123601315485785 |
| 10 | 0.05 | 2 | 4.127443443454726 |
| 5 | 0.05 | 3 | 4.140278255258786 |
| 5 | 0.05 | 2 | 4.16734798202403 |
| 5 | 0.05 | 5 | 4.209423355616378 |
| 10 | 0.05 | 5 | 4.220276266031 |
| 10 | 0.05 | 7 | 4.248097378547858 |
| 5 | 0.05 | 7 | 4.254838537157542 |
| 5 | 0.05 | 10 | 4.279299834462245 |
| 10 | 0.05 | 10 | 4.2867768798705255 |

---


## Conclusion

So, should you prime your embedding space first? **Maybe.**

*   If you are using a simple **word-level vocabulary**, this experiment suggests the answer is a strong **yes**. A short, corpus-specific pre-training phase can provide a substantial performance boost, provided you tune the pre-training hyperparameters carefully.

*   If you are using a more advanced **subword tokenizer like BPE**, the benefits appear to be **less pronounced but still positive**. The inherent advantages of the tokenizer itself might already solve some of the problems that embedding priming aims to fix.

In either case, this technique represents a trade-off: a small increase in upfront training complexity and time in exchange for a potentially faster and better convergence of your main model. It's a tool worth having in your toolkit and exploring for your specific task and dataset.
