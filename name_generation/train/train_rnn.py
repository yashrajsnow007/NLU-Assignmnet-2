import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rnn import VanillaRNN
from utils.preprocess import load_data, get_loader
from utils.evaluation import evaluate

# Hyperparameters for vanilla RNN training
EMBED_SIZE = 32
HIDDEN_SIZE = 128
LEARNING_RATE = 0.003
EPOCHS = 25 

def generate(model, stoi, itos, device, max_len=20):
    # Generate name by sampling character-by-character
    model.eval()

    # Start with '<' token
    char = torch.tensor([[stoi['<']]]).to(device)
    name = ""

    # Initialize hidden state
    h = torch.zeros(1, model.hidden_size).to(device)

    # Generate characters until '>' or max_len
    for _ in range(max_len):
        # Get embedding for current character
        x = model.embedding(char[:, -1])

        # Compute RNN hidden state
        h = torch.tanh(
            x @ model.W_xh +
            h @ model.W_hh +
            model.b_h
        )

        # Get logits for next character
        logits = (h @ model.W_hy + model.b_y)

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
    model = VanillaRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)

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
            # Forward pass
            out = model(x)

            # Compute loss and backprop
            loss = criterion(out.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "train/rnn_model.pth")
    print("Trainable Parameters:", count_params(model))

    # Evaluate model on novelty and diversity
    novelty, diversity, _ = evaluate(
        lambda m: generate(m, stoi, itos, device),
        model,
        names,
        save_path="results/rnn.txt"
    )

    print("Novelty:", novelty)
    print("Diversity:", diversity)


if __name__ == "__main__":
    train()