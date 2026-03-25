"""
Multi-Layer Perceptron (MLP) architecture for text classification.
"""

import torch
import torch.nn as nn
from src.preprocess import PAD # Import the PAD constant from our preprocessing script

class MLPClassifier(nn.Module):
    """
    MLP classifier that aggregates word embeddings using mean-pooling.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.3):
        """
        Initializes the embedding layer and the dynamic MLP hidden layers.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimensionality of the word embeddings.
            hidden_dims (list[int]): List containing the sizes of each hidden layer.
            num_classes (int): Number of output classes (e.g., 2 for binary).
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        
        # Embedding layer with padding index specified so PAD tokens are zeroed out
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        
        # Build hidden layers dynamically based on the hidden_dims list
        layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
            
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP.
        
        Args:
            x (torch.Tensor): Batch of tokenized sequences of shape (B, T).
            
        Returns:
            torch.Tensor: Raw logit predictions of shape (B, num_classes).
        """
        # x: (Batch_Size, Sequence_Length)
        emb = self.embedding(x) # Shape: (B, T, embed_dim)
        
        # Create a mask to ignore padding tokens during mean pooling
        mask = (x != PAD).unsqueeze(2) # Shape: (B, T, 1)
        emb = emb * mask 
        
        # Mean pool over the non-padding tokens
        # Sum embeddings across the sequence length, then divide by the actual number of words
        pooled = emb.sum(1) / mask.sum(1).clamp(min=1) # Shape: (B, embed_dim)
        
        # Pass the mean-pooled sentence representation through the MLP
        return self.classifier(pooled)