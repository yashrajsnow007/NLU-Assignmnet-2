import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Load names and prepare character-level data
def load_data(path):
    # Read names from file, lowercase
    with open(path, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    
    # Add start '<' and end '>' markers to mark sequence boundaries
    names = ["<" + name + ">" for name in names]
    
    # Build vocabulary: unique characters in dataset
    chars = sorted(list(set("".join(names))))
    stoi = {ch: i for i, ch in enumerate(chars)}  # char -> index
    itos = {i: ch for ch, i in stoi.items()}       # index -> char
    
    # Encode each name as sequence of character indices
    encoded = [[stoi[ch] for ch in name] for name in names]
    
    # Create (input, target) training pairs by shifting sequences by 1 position
    data = [(seq[:-1], seq[1:]) for seq in encoded]
    
    return data, stoi, itos, len(chars), names


def collate_fn(batch):
    # Extract inputs and targets from batch tuples
    x = [torch.tensor(i) for i, j in batch]
    y = [torch.tensor(j) for i, j in batch]
    
    # Pad sequences to same length (padding value=0)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    
    return x, y


def get_loader(data, batch_size=32):
    # Create DataLoader with custom collation for padded sequences
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
