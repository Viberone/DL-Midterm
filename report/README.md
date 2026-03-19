# IMDB Sentiment Analysis: MLP vs RNN Comparison

## Project Overview

This project implements binary sentiment classification on the IMDB dataset, comparing two neural network architectures:
- **MLP (Multi-Layer Perceptron)**: Mean-pooled embeddings with fully-connected layers
- **RNN variants (Vanilla RNN, LSTM, GRU)**: Sequential models with recurrent connections

The coursework spans **Weeks 7-9** with controlled experiments comparing network depth, embedding dimensions, dropout rates, and architectural choices.

## Dataset

- **Source**: IMDB Reviews (Hugging Face Datasets)
- **Train/Val/Test Split**: 70% / 10% / 20%
- **Vocabulary Size**: 20,000 most frequent words
- **Sequence Length**: Max 256 tokens (dynamically padded to batch max)
- **Classes**: Binary (0=NEGATIVE, 1=POSITIVE)

**Statistics:**
- Total reviews: 35,000 (25K labeled train + 25K test)
- Class distribution: Balanced (12.5K positive, 12.5K negative per split)
- Sequence lengths: Min 10, Max 2,494, Mean 237 tokens

## Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **MLP** | 49.24% | 0.589 | 0.013 | 0.025 |
| **RNN (GRU)** | **92.82%** | **0.922** | **0.938** | **0.930** |

**Winner**: RNN outperforms MLP by **43.58%**

## Installation & Setup

**Requirements:**
- Python 3.12+
- PyTorch 2.10.0 with CUDA support
- See `requirements.txt` for dependencies

**Setup:**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

## Project Structure

- **notebooks/**: 4 Jupyter notebooks (Parts A-D)
  - 01_eda.ipynb: Exploratory Data Analysis
  - 02_mlp.ipynb: MLP experiments (3 controlled tests)
  - 03_rnn.ipynb: RNN variants (Vanilla/LSTM/GRU)
  - 04_analysis.ipynb: Model comparison & error analysis

- **src/**: Core implementation modules
  - preprocess.py: Data pipeline
  - mlp_model.py: MLP architecture
  - rnn_mode.py: RNN/LSTM/GRU architecture
  - train.py: Shared training loop
  - evaluate.py: Metrics calculation

- **checkpoints/**: Trained model weights
  - mlp_best.pt: Best MLP (256-dim embed, [128,128] hidden)
  - rnn_best.pt: Best RNN/GRU (2 layers, 256-dim embed)

## Running the Project

```bash
# Start Jupyter and run notebooks in order:
jupyter notebook

# Notebooks should be executed in sequence:
# 1. 01_eda.ipynb (~30 sec)
# 2. 02_mlp.ipynb (~2-3 min CPU, ~10 sec GPU)
# 3. 03_rnn.ipynb (~15-20 min GPU, 2+ hours CPU)
# 4. 04_analysis.ipynb (~1 min)
```

## Key Findings

**MLP Weaknesses:**
- Treats text as bag-of-words (ignores word order)
- Cannot capture negation effects ("not bad" vs "bad not")
- Poor on sarcasm and context-dependent sentiment

**RNN Strengths:**
- Captures sequential context and dependencies
- GRU's gating mechanisms handle vanishing gradients
- Multiple layers extract multi-level features
- 2,378 cases where RNN succeeds but MLP fails

## References

1. Maas et al. (2011). "Learning Word Vectors for Sentiment Analysis". ACL.
2. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory". Neural Computation.
3. Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder". EMNLP.

---

**Status**: Complete (Parts A-D)  
**Last Updated**: March 2026
