import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blstm import BLSTM
from utils.preprocess import load_data, get_loader
from utils.evaluation import evaluate


# 🔹 Hyperparameters
EMBED_SIZE = 32
HIDDEN_SIZE = 128
LEARNING_RATE = 0.003
EPOCHS = 25  

def generate(model, stoi, itos, device, max_len=20):
    model.eval()

    char = torch.tensor([[stoi['<']]]).to(device)
    name = ""

    h = torch.zeros(1, model.hidden_size).to(device)
    c = torch.zeros(1, model.hidden_size).to(device)

    for _ in range(max_len):
        x = model.embedding(char[:, -1])

        h, c = model.forward_cell(x, h, c)

        logits = model.fc_forward(h)

        # prevent '<'
        logits[0, stoi['<']] = -1e9

        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1)

        ch = itos[next_char.item()]
        if ch == '>':
            break

        name += ch
        char = next_char

    return name


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    data, stoi, itos, vocab_size, names = load_data("data/TrainingNames.txt")
    loader = get_loader(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs, forward_logits = model(x)

            loss1 = criterion(outputs.view(-1, vocab_size), y.view(-1))
            loss2 = criterion(forward_logits.view(-1, vocab_size), y.view(-1))

            loss = loss1 + loss2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("Trainable Parameters:", count_params(model))

    novelty, diversity, _ = evaluate(
        lambda m: generate(m, stoi, itos, device),
        model,
        names,
        save_path="results/attention.txt"
    )

    print("Novelty:", novelty)
    print("Diversity:", diversity)


if __name__ == "__main__":
    train()