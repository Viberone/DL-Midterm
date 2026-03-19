"""
Recurrent Neural Network architectures (Vanilla RNN, LSTM, GRU) for text classification.
"""

import torch
import torch.nn as nn
from src.preprocess import PAD

class RNNClassifier(nn.Module):
    """
    A flexible Recurrent Neural Network classifier that supports Vanilla RNN, LSTM, and GRU.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 num_layers: int, num_classes: int, rnn_type: str = "lstm", 
                 dropout: float = 0.3):
        """
        Initializes the embedding layer, recurrent layers, and classification head.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the word embeddings.
            hidden_dim (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            num_classes (int): Number of output classes.
            rnn_type (str): Type of RNN to use ('rnn', 'lstm', or 'gru').
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        
        # Map the string argument to the actual PyTorch module
        rnn_cls = {
            "rnn": nn.RNN, 
            "lstm": nn.LSTM, 
            "gru": nn.GRU
        }[rnn_type.lower()]
        
        # Initialize the chosen recurrent layer
        self.rnn = rnn_cls(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.rnn_type = rnn_type.lower()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RNN classifier using packed sequences.
        
        Args:
            x (torch.Tensor): Batch of tokenized sequences of shape (B, T).
            lengths (torch.Tensor): 1D tensor of valid sequence lengths for each item in the batch.
            
        Returns:
            torch.Tensor: Raw logit predictions of shape (B, num_classes).
        """
        # Apply dropout to embeddings as a regularization technique
        emb = self.dropout(self.embedding(x))  # Shape: (B, T, d_e)
        
        # Pack the sequences to ignore padding tokens during recurrent processing
        # enforce_sorted=False allows us to pass batches that aren't sorted by length
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through the recurrent layer
        out, hidden = self.rnn(packed)
        
        # Extract the hidden state of the final layer to use for classification
        if self.rnn_type == "lstm":
            # LSTM returns a tuple: (h_n, c_n). We only want the hidden state (h_n).
            # h_n shape: (num_layers * num_directions, batch, hidden_size)
            h_last = hidden[0][-1] 
        else:
            # GRU and Vanilla RNN just return h_n
            h_last = hidden[-1]
            
        return self.fc(self.dropout(h_last))