import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.W_xh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)

        h = torch.zeros(B, self.hidden_size).to(x.device)
        hidden_states = []

        for t in range(T):
            h = torch.tanh(x[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h.unsqueeze(1))

        hidden_states = torch.cat(hidden_states, dim=1)

        outputs = []

        for t in range(T):
            h_t = hidden_states[:, t, :]

            scores = torch.bmm(hidden_states, h_t.unsqueeze(2)).squeeze(2)
            attn_weights = F.softmax(scores, dim=1)

            context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)

            combined = torch.cat([h_t, context], dim=1)
            out = self.fc(combined)

            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)