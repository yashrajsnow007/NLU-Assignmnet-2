class BLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.forward_cell = LSTMCell(embed_size, hidden_size)
        self.backward_cell = LSTMCell(embed_size, hidden_size)
        
        # bidirectional output
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        
        # forward-only output (IMPORTANT)
        self.fc_forward = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        x = self.embedding(x)
        
        # forward pass
        h_f = torch.zeros(B, self.hidden_size).to(x.device)
        c_f = torch.zeros(B, self.hidden_size).to(x.device)
        
        forward_outputs = []
        for t in range(T):
            h_f, c_f = self.forward_cell(x[:, t, :], h_f, c_f)
            forward_outputs.append(h_f.unsqueeze(1))
        
        forward_outputs = torch.cat(forward_outputs, dim=1)
        
        # backward pass
        h_b = torch.zeros(B, self.hidden_size).to(x.device)
        c_b = torch.zeros(B, self.hidden_size).to(x.device)
        
        backward_outputs = []
        for t in reversed(range(T)):
            h_b, c_b = self.backward_cell(x[:, t, :], h_b, c_b)
            backward_outputs.append(h_b.unsqueeze(1))
        
        backward_outputs.reverse()
        backward_outputs = torch.cat(backward_outputs, dim=1)
        
        # BLSTM output
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2)
        outputs = self.fc(outputs)
        
        # 🔴 forward-only output (THIS FIXES EVERYTHING)
        forward_logits = self.fc_forward(forward_outputs)
        
        return outputs, forward_logits