import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def load_data(path):
    with open(path, "r") as f:
        names = [line.strip().lower() for line in f if line.strip()]
    
    names = ["<" + name + ">" for name in names]
    
    chars = sorted(list(set("".join(names))))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    
    encoded = [[stoi[ch] for ch in name] for name in names]
    
    data = [(seq[:-1], seq[1:]) for seq in encoded]
    
    return data, stoi, itos, len(chars), names


def collate_fn(batch):
    x = [torch.tensor(i) for i, j in batch]
    y = [torch.tensor(j) for i, j in batch]
    
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    
    return x, y


def get_loader(data, batch_size=32):
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
