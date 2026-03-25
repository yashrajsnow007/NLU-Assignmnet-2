import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN with attention mechanism for character-level name generation
class RNNAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # RNN weights (input-to-hidden, hidden-to-hidden, bias)
        self.W_xh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))

        # Output layer: projects concatenated [hidden, context] to vocabulary
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        B, T = x.shape
        # Convert character indices to embeddings
        x = self.embedding(x)

        # Encode all hidden states first
        h = torch.zeros(B, self.hidden_size).to(x.device)
        hidden_states = []

        # Process sequence and collect all hidden states
        for t in range(T):
            h = torch.tanh(x[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h.unsqueeze(1))

        # Stack all hidden states for attention computation
        hidden_states = torch.cat(hidden_states, dim=1)

        outputs = []

        # Generate output for each position using attention over all hidden states
        for t in range(T):
            h_t = hidden_states[:, t, :]

            # Compute attention scores: how much to attend to each hidden state
            scores = torch.bmm(hidden_states, h_t.unsqueeze(2)).squeeze(2)
            # Normalize attention weights using softmax
            attn_weights = F.softmax(scores, dim=1)

            # Compute context vector as weighted sum of hidden states
            context = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)

            # Concatenate current hidden state with context vector
            combined = torch.cat([h_t, context], dim=1)
            # Project to vocabulary logits
            out = self.fc(combined)

            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)