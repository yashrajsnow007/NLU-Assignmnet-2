import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blstm import BLSTM
from utils.preprocess import load_data, get_loader
from utils.evaluation import evaluate

# Hyperparameters for Bidirectional LSTM training
EMBED_SIZE = 32
HIDDEN_SIZE = 128
LEARNING_RATE = 0.003
EPOCHS = 25


def generate(model, stoi, itos, device, max_len=20):
    # Generate name using forward LSTM cell for inference
    model.eval()

    # Start with '<' token
    char = torch.tensor([[stoi['<']]]).to(device)
    name = ""

    # Initialize hidden and cell states
    h = torch.zeros(1, model.hidden_size).to(device)
    c = torch.zeros(1, model.hidden_size).to(device)

    # Generate characters until '>' or max_len
    for _ in range(max_len):
        # Get embedding for current character
        x = model.embedding(char[:, -1])

        # Compute forward LSTM hidden state
        h, c = model.forward_cell(x, h, c)

        # Get logits for next character from forward cell only
        logits = model.fc_forward(h)

        # Prevent start token '<' from being generated
        logits[0, stoi['<']] = -1e9

        # Sample next character from probability distribution
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)

        # Check if end token reached
        ch = itos[next_char.item()]
        if ch == '>':
            break

        name += ch
        char = next_char

    return name

def count_params(model):
    # Count total trainable parameters in model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    # Load training data
    data, stoi, itos, vocab_size, names = load_data("data/TrainingNames.txt")
    loader = get_loader(data)

    # Use GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and move to device
    model = BLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)

    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padded tokens

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        # Iterate through batches
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Forward pass: get outputs from bidirectional and forward-only paths
            outputs, forward_logits = model(x)

            # Compute loss for both outputs (auxiliary loss)
            loss1 = criterion(outputs.view(-1, vocab_size), y.view(-1))
            loss2 = criterion(forward_logits.view(-1, vocab_size), y.view(-1))

            # Combine both losses
            loss = loss1 + loss2

            # Backprop with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "train/blstm_model.pth")

    print("Trainable Parameters:", count_params(model))
    # Evaluate model on novelty and diversity
    novelty, diversity, _ = evaluate(
        lambda m: generate(m, stoi, itos, device),
        model,
        names,
        save_path="results/blstm.txt"
    )

    
    print("Novelty:", novelty)
    print("Diversity:", diversity)


if __name__ == "__main__":
    train()