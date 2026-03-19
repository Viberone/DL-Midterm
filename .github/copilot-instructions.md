# Copilot Instructions for Midterm_DL

## Project Overview
**Sentiment Analysis on IMDB Dataset**: Binary classification task (positive/negative) comparing MLP and RNN architectures. Coursework spanning Weeks 7-9 with specific deliverables: preprocessing pipeline, MLP classifier, RNN variants (LSTM/GRU), controlled experiments, and technical report. All work runs through Jupyter notebooks.

## Architecture

### Data Pipeline (`src/preprocess.py`)
- **Vocabulary building**: Frequency-based word-to-ID mapping (special tokens: PAD=0, UNK=1)
- **Text cleaning**: Removes HTML tags, keeps only letters and spaces, lowercases
- **Dynamic padding**: `collate_fn()` pads sequences to batch-max length (not fixed max_len), improving efficiency
- **Key pattern**: Words not in vocab map to UNK token via `vocab.get(w, UNK)`, sequences truncate at `max_len=256`

### Model Architectures

**MLP Classifier** (`src/mlp_model.py`):
- Embeds word tokens → masks PAD tokens → mean-pools over sequence → passes through dynamic hidden layers
- Masking prevents padding from distorting embeddings: `mask = (x != PAD).unsqueeze(2)` then `emb = emb * mask`
- Hidden layers configurable via list: `hidden_dims=[64, 32]` creates [Linear(embed_dim→64), ReLU, Dropout] + [Linear(64→32), ReLU, Dropout]
- **Call signature**: `model(inputs)` — no sequence lengths needed

**RNN Classifier** (`src/rnn_mode.py`):
- Supports vanilla RNN, LSTM, GRU via string parameter `rnn_type`
- Uses `pack_padded_sequence` with `enforce_sorted=False` for efficient batch processing
- Extracts final hidden state: for LSTM uses `hidden[0][-1]` (h_n, ignore c_n), for RNN/GRU uses `hidden[-1]`
- **Call signature**: `model(inputs, lengths)` — requires sequence lengths calculated as `(inputs != 0).sum(dim=1).clamp(min=1)`

### Training & Evaluation (`src/train.py`, `src/evaluate.py`)
- **Shared `train_epoch()`**: Flag `is_rnn=True/False` determines if model receives lengths
- **Evaluation pattern**: Disabled model dropout, run on test set, collect predictions via `torch.max(outputs, 1)`
- **Metrics**: Accuracy, Precision, Recall, F1 via scikit-learn with `average='binary'` for 2-class classification

## Development Workflow

### Notebook Structure (Required for Submission)
The project requires **4 notebooks** (order of execution):

1. **`01_eda.ipynb`** (Part A, ~Week 7)
   - Load IMDB dataset via `load_dataset("imdb")` → get 25K train, 25K test, 50K unsupervised splits
   - **Required EDA outputs**: class distribution (balanced binary), sequence length stats (min/max/mean/90th percentile), vocabulary frequency bar chart (top-30 tokens), representative samples
   - **Purpose**: Informs choice of `max_len` for padding and vocabulary size

2. **`02_mlp.ipynb`** (Part B, ~Week 7)
   - Single end-to-end cell: load data → build vocab → create SentimentDataset → train MLP for 5 epochs
   - **3 required controlled experiments**:
     - Network depth: 1 vs 2 vs 3 hidden layers (keep parameter count ~constant)
     - Embedding dimension: 64, 128, 256 (use best architecture from depth experiment)
     - Dropout rate: 0.2, 0.3, 0.5 (use best configs from dims experiment)
   - Record all metrics (Accuracy, Precision, Recall, F1, Time/epoch) for final comparison table

3. **`03_rnn.ipynb`** (Part C, ~Week 8)
   - **Implement all 3 variants**: Vanilla RNN, LSTM, GRU using native PyTorch (`nn.RNN`, `nn.LSTM`, `nn.GRU`)
   - **3 required controlled experiments**:
     - RNN variant comparison: Vanilla vs LSTM vs GRU (fixed: `embed_dim=128, hidden_dim=128, num_layers=1`)
     - Embedding dimension: 64, 128, 256 (use best variant from first experiment)
     - Number of layers: 1 vs 2 (use best dims from second experiment)
   - **Critical**: Use `pack_padded_sequence` with `enforce_sorted=False` for correct padding handling
   - Extract final hidden state: LSTM takes `hidden[0][-1]` (tuple unpacking), RNN/GRU take `hidden[-1]`

4. **`04_analysis.ipynb`** (Part D, ~Week 9)
   - Load checkpoint files: `checkpoints/mlp_best.pt`, `checkpoints/rnn_best.pt`
   - **Generate summary table**: Model, Accuracy, Precision, Recall, F1, Time/epoch (all 5 configs: best MLP + 4 RNN variants)
   - **Plot learning curves**: Loss and accuracy on both train/val for ≥2 models with noticeably different behavior
   - **Qualitative error analysis**: 5-10 misclassified test examples from best model—identify linguistic patterns (negation handling, sarcasm, domain mismatch)
   - **Answer required questions**:
     1. Which model wins? Theoretical vs observed?
     2. Overfitting evidence? Mitigation effectiveness?
     3. MLP blind to word order—concrete counter-example from test set?

### Expected Outcomes
- **Week 7**: EDA notebook + working MLP with baseline results (~85-90% accuracy, ~1-2 min/epoch)
- **Week 8**: 3 trained RNN variants + preliminary results table + learning curves
- **Week 9**: Polished analysis, error examples, technical report (~15 pages, APA format)

## Common Tasks & Patterns

### Running Controlled Experiments
**Pattern**: Modify one hyperparameter at a time, keep others fixed. Example for MLP depth:

```python
# Experiment 1: Network Depth (keep total params ~50K)
configs = [
    {"hidden_dims": [256], "embed_dim": 128},         # 1 hidden layer
    {"hidden_dims": [128, 128], "embed_dim": 128},    # 2 hidden layers
    {"hidden_dims": [85, 85, 85], "embed_dim": 128},  # 3 hidden layers (adjusted sizes)
]
results = {}
for cfg in configs:
    model = MLPClassifier(len(vocab), cfg["embed_dim"], cfg["hidden_dims"], 2, 0.3)
    # Train 5 epochs, record all metrics
    results[str(cfg)] = {"acc": ..., "prec": ..., "recall": ..., "f1": ..., "time": ...}
```

**Record format** (for final report table):
| Model | Acc | Prec | Recall | F1 | Time/epoch |
|-------|-----|------|--------|----|----|

### Saving & Loading Best Models
- Always save after training: `torch.save(model.state_dict(), f"checkpoints/{model_type}_best.pt")`
- Load checkpoint: `model.load_state_dict(torch.load("checkpoints/mlp_best.pt")); model.eval()`
- Use `.eval()` mode before any inference (disables dropout, batch norm)

### Extracting RNN Hidden States (Critical)
```python
# LSTM returns (output, (h_n, c_n)) — tuple of 2 tensors
out, hidden = self.rnn(packed)  # hidden[0] = h_n, hidden[1] = c_n
h_last = hidden[0][-1]  # Take h_n (hidden[1] is cell state, ignored)

# Vanilla RNN / GRU return (output, h_n) — just 1 tensor
out, hidden = self.rnn(packed)  # hidden already = h_n
h_last = hidden[-1]  # Last layer's hidden state
```

### Debugging Sequence Length Issues
```python
# ✓ Correct: compute on GPU, then move to CPU before pack_padded_sequence
lengths = (inputs != 0).sum(dim=1).clamp(min=1)  # GPU tensor
lengths_cpu = lengths.cpu()  # Move to CPU
packed = nn.utils.rnn.pack_padded_sequence(emb, lengths_cpu, batch_first=True, enforce_sorted=False)

# ✗ Wrong: pass GPU tensor directly to pack_padded_sequence
packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, ...)  # Will error
```

### Plotting Learning Curves
```python
import matplotlib.pyplot as plt
# Save metrics during training: train_losses, train_accs, val_losses, val_accs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train"); ax1.plot(val_losses, label="Val")
ax1.set_ylabel("Loss"); ax1.legend()
ax2.plot(train_accs, label="Train"); ax2.plot(val_accs, label="Val")
ax2.set_ylabel("Accuracy"); ax2.legend()
plt.show()
```

### Adding a New Model
1. Create class inheriting `nn.Module` in new file (`src/new_model.py`)
2. Implement `forward()` to accept `(inputs)` or `(inputs, lengths)` depending on architecture
3. Update `train_epoch()` conditional if using new pattern, or pass `is_rnn=True/False`

## Report Requirements (Submission Checklist)

### Technical Report (~15 pages, APA format, Times New Roman 12pt, 1.5 spacing)
1. **Abstract** (≤200 words): Problem, approach, main findings
2. **Dataset Description**: IMDB statistics, class distribution, sequence length histogram, EDA highlights
3. **Methods**: MLP architecture diagram, RNN equations (Vanilla/LSTM/GRU), training details (loss, optimizer, learning rate)
4. **Experiments**: Table of all 5 configs + results, learning curves for ≥2 diverse models, experimental protocol
5. **Analysis & Discussion**:
   - Which model wins? Explain theoretically and empirically
   - Evidence of overfitting? Mitigation strategies?
   - **MLP blind to word order**: Provide 1-2 concrete test examples where RNN succeeds but MLP fails
   - Error analysis: 5-10 misclassified examples, identify patterns (negation, sarcasm, rare words)
6. **Conclusion**: Summary, limitations, future directions
7. **References**: ≥4 sources (APA or IEEE format)

### Code Requirements
- **All notebooks must run end-to-end** without errors (Restart & Run All)
- Every function and class must have a **concise docstring** (inputs, outputs, brief description)
- `README.md` must specify:
  - Team member contributions (if applicable)
  - Environment setup: `pip install -r requirements.txt`
  - Command to reproduce experiments (or "Run all cells in notebooks/ in order: 01 → 02 → 03 → 04")

### Submission Structure (Single .zip file)
```
ID1_ID2_Midterm/
├── notebooks/
│   ├── 01_eda.ipynb           # EDA, dataset exploration
│   ├── 02_mlp.ipynb           # MLP training + experiments
│   ├── 03_rnn.ipynb           # RNN/LSTM/GRU training + experiments
│   └── 04_analysis.ipynb      # Final comparison, error analysis, answers to questions
├── src/
│   ├── preprocess.py          # Data pipeline (provided)
│   ├── mlp_model.py           # MLPClassifier (provided)
│   ├── rnn_model.py           # RNNClassifier (provided)
│   ├── train.py               # Shared training loop (provided)
│   └── evaluate.py            # Metrics calculation (provided)
├── checkpoints/
│   ├── mlp_best.pt            # Best MLP weights
│   └── rnn_best.pt            # Best RNN weights
├── report/
│   └── report.pdf             # Final technical report
├── requirements.txt           # Python dependencies
└── README.md                  # Setup & reproduction instructions
```

## Key Design Decisions

- **Dynamic padding over fixed**: Reduces wasted computation vs padding all sequences to 256
- **PAD/UNK constants in preprocess.py**: Single source of truth (imported by models to use `padding_idx=PAD`)
- **Flexible RNN types**: String parameter `rnn_type` allows comparing architectures without code duplication
- **Binary classification with softmax**: Cross-entropy loss expects class logits (not probabilities)

## Gotchas

1. **Sequence lengths for RNN**: Must be computed from input tensors before moving to device: `lengths = (inputs != 0).sum(dim=1)` then `.cpu()` before packing
2. **LSTM hidden state unpacking**: `hidden` is tuple `(h_n, c_n)` — only take `hidden[0]` for final layer output
3. **Vocab OOV handling**: Unknown words silently map to UNK (ID=1); not an error, by design
4. **Batch-size dependent metrics**: Dropout and batch norm behave differently in train vs eval mode — always call `model.eval()` before inference
