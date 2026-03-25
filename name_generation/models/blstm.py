import torch
import torch.nn as nn

# LSTM cell implementation for bidirectional architecture
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Forget gate weights (input and hidden)
        self.W_f = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        # Input gate weights
        self.W_i = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        # Candidate gate weights
        self.W_g = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        # Output gate weights
        self.W_o = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)

        # Recurrent weights for each gate
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.U_g = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

        # Biases for each gate
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h, c):
        # Compute LSTM gates (forget, input, candidate, output)
        f = torch.sigmoid(x @ self.W_f + h @ self.U_f + self.b_f)
        i = torch.sigmoid(x @ self.W_i + h @ self.U_i + self.b_i)
        g = torch.tanh(x @ self.W_g + h @ self.U_g + self.b_g)
        o = torch.sigmoid(x @ self.W_o + h @ self.U_o + self.b_o)

        # Update cell state and hidden state
        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


# Bidirectional LSTM for character-level name generation
class BLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Forward and backward LSTM cells for bidirectional processing
        self.forward_cell = LSTMCell(embed_size, hidden_size)
        self.backward_cell = LSTMCell(embed_size, hidden_size)

        # Output layers: combined bidirectional and forward only
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.fc_forward = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        # Convert character indices to embeddings
        x = self.embedding(x)

        # Initialize forward LSTM hidden and cell states
        h_f = torch.zeros(B, self.hidden_size).to(x.device)
        c_f = torch.zeros(B, self.hidden_size).to(x.device)

        # Process sequence forward
        forward_outputs = []
        for t in range(T):
            h_f, c_f = self.forward_cell(x[:, t, :], h_f, c_f)
            forward_outputs.append(h_f.unsqueeze(1))

        forward_outputs = torch.cat(forward_outputs, dim=1)

        # Initialize backward LSTM hidden and cell states
        h_b = torch.zeros(B, self.hidden_size).to(x.device)
        c_b = torch.zeros(B, self.hidden_size).to(x.device)

        # Process sequence backward (reverse order)
        backward_outputs = []
        for t in reversed(range(T)):
            h_b, c_b = self.backward_cell(x[:, t, :], h_b, c_b)
            backward_outputs.append(h_b.unsqueeze(1))

        # Reverse backward outputs to match forward sequence order
        backward_outputs.reverse()
        backward_outputs = torch.cat(backward_outputs, dim=1)

        # Concatenate forward and backward hidden states
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)
        # Project concatenated states to vocabulary
        outputs = self.fc(outputs)

        # Also output forward-only logits for auxiliary loss
        forward_logits = self.fc_forward(forward_outputs)

        return outputs, forward_logits