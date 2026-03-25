import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.W_xh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))
        
        self.W_hy = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_y  = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)
        
        h = torch.zeros(B, self.hidden_size).to(x.device)
        outputs = []
        
        for t in range(T):
            h = torch.tanh(x[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
            y = h @ self.W_hy + self.b_y
            outputs.append(y.unsqueeze(1))
        
        return torch.cat(outputs, dim=1)
