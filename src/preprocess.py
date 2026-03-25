"""
Data preprocessing pipeline for Sentiment Analysis.
Handles text cleaning, tokenization, vocabulary construction, and dataset formatting.
"""

from collections import Counter
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Define special token constants
PAD, UNK = 0, 1

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags and non-alphabetical characters.
    
    Args:
        text (str): The raw input text string.
        
    Returns:
        str: Lowercased string containing only letters and spaces.
    """
    text = re.sub(r"<[^>]+>", " ", text)        # Strip HTML tags like <br />
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # Keep letters and spaces only
    return text.lower().strip()

def build_vocab(texts: list[str], max_size: int = 20000) -> dict[str, int]:
    """
    Builds a vocabulary mapping from words to integer IDs based on frequency.
    
    Args:
        texts (list[str]): List of training sentences/reviews.
        max_size (int): Maximum vocabulary size, including special tokens.
        
    Returns:
        dict[str, int]: A dictionary mapping tokens to their integer IDs.
    """
    # Count frequencies of all words in the cleaned texts
    counter = Counter(w for t in texts for w in clean_text(t).split())
    
    # Initialize vocabulary with special padding and unknown tokens
    vocab = {"<pad>": PAD, "<unk>": UNK}
    
    # Add the most common words up to the max_size limit
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
        
    return vocab

class SentimentDataset(Dataset):
    """
    PyTorch Dataset for converting text sequences to padded integer tensors.
    """
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int = 256):
        """
        Initializes the dataset, mapping words to IDs and truncating/padding.
        
        Args:
            texts (list[str]): List of text samples.
            labels (list[int]): List of corresponding integer labels.
            vocab (dict[str, int]): Vocabulary mapping dictionary.
            max_len (int): Maximum sequence length for truncation.
        """
        self.data = []
        for text, label in zip(texts, labels):
            # Convert words to IDs; use UNK if word is not in vocab
            ids = [vocab.get(w, UNK) for w in clean_text(text).split()]
            ids = ids[:max_len] # Truncate sequences longer than max_len
            
            # Store as PyTorch tensors
            self.data.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(label, dtype=torch.long)
            ))

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetches the (sequence_tensor, label_tensor) at index i."""
        return self.data[i]

def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for the DataLoader to dynamically pad batches.
    
    Args:
        batch: List of tuples (sequence, label) returned by Dataset.__getitem__.
        
    Returns:
        tuple containing padded sequences tensor and stacked labels tensor.
    """
    seqs, labels = zip(*batch)
    # Pad sequences to the length of the longest sequence in this specific batch
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    return seqs_padded, torch.stack(labels)