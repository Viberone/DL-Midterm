"""
Evaluation metrics script for calculating Precision, Recall, F1, and Accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device, is_rnn: bool = False) -> tuple[list[int], list[int]]:
    """
    Runs inference over a dataloader and collects all true and predicted labels.
    
    Args:
        model (nn.Module): The trained model to evaluate.
        dataloader (DataLoader): DataLoader for the test or validation set.
        device (torch.device): CPU or CUDA device.
        is_rnn (bool): Flag indicating if the model needs sequence lengths.
        
    Returns:
        tuple: (list of true labels, list of predicted labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            if is_rnn:
                lengths = (inputs != 0).sum(dim=1).clamp(min=1)
                outputs = model(inputs, lengths)
            else:
                outputs = model(inputs)
                
            _, predicted = torch.max(outputs, 1)
            
            # Move back to CPU and convert to lists for scikit-learn
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """
    Calculates Accuracy, Precision, Recall, and F1-score.
    
    Args:
        y_true (list[int]): Ground truth labels.
        y_pred (list[int]): Predicted labels from the model.
        
    Returns:
        dict[str, float]: Dictionary containing the calculated metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # We use average='binary' since this is a positive/negative classification task
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }