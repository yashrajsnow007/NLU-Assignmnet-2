import torch
import torch.nn as nn

# Vanilla RNN for character-level name generation
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # RNN weights: input-to-hidden, hidden-to-hidden, and biases
        self.W_xh = nn.Parameter(torch.randn(embed_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))
        
        # Output layer weights: hidden-to-vocabulary
        self.W_hy = nn.Parameter(torch.randn(hidden_size, vocab_size) * 0.01)
        self.b_y  = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, x):
        B, T = x.shape  # batch_size, sequence_length
        # Convert character indices to embeddings
        x = self.embedding(x)
        
        # Initialize hidden state to zeros
        h = torch.zeros(B, self.hidden_size).to(x.device)
        outputs = []
        
        # Process each timestep sequentially (unrolled RNN)
        for t in range(T):
            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            h = torch.tanh(x[:, t, :] @ self.W_xh + h @ self.W_hh + self.b_h)
            # Project hidden state to output logits
            y = h @ self.W_hy + self.b_y
            outputs.append(y.unsqueeze(1))
        
        # Concatenate all timestep outputs
        return torch.cat(outputs, dim=1)
