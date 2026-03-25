"""
Shared training and validation loops for all models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: torch.device, 
                is_rnn: bool = False) -> tuple[float, float]:
    """
    Trains the model for a single epoch.
    
    Args:
        model (nn.Module): The model to train (MLP or RNN).
        dataloader (DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimization algorithm (e.g., Adam).
        device (torch.device): CPU or CUDA device.
        is_rnn (bool): Flag to indicate if the model needs sequence lengths (RNN) or not (MLP).
        
    Returns:
        tuple: (average_loss, average_accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # RNN requires sequence lengths for pack_padded_sequence
        if is_rnn:
            # Calculate lengths by finding non-pad tokens (assuming PAD is 0)
            lengths = (inputs != 0).sum(dim=1)
            # Clamp lengths to a minimum of 1 to avoid errors on completely empty sequences
            lengths = lengths.clamp(min=1)
            outputs = model(inputs, lengths)
        else:
            outputs = model(inputs)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / total, correct / total