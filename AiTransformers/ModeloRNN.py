import os
import math
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Config
CORPUS_PATH = "C:\\Users\\ViniciusLima\\Documents\\UFRPE_PROJETOS\\AI4GOOD\\Atividade2\\Ai4Good-GPT-2-\\AiTransformers\\corpus_clean.txt"
SEQ_LEN = 30
BATCH_SIZE = 64
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
LR = 1e-3
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "rnn_lm.pt"
MIN_FREQ = 2  # ignore rare tokens

# Simple whitespace tokenizer
def read_corpus(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tokens = text.split()
    return tokens

def build_vocab(tokens, min_freq=1):
    ctr = Counter(tokens)
    vocab = ["<pad>", "<unk>"]
    for tok, c in ctr.most_common():
        if c >= min_freq:
            vocab.append(tok)
    stoi = {s:i for i,s in enumerate(vocab)}
    return vocab, stoi

class SeqDataset(Dataset):
    def __init__(self, ids, seq_len):
        self.ids = ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + self.seq_len + 1], dtype=torch.long)
        return x, y

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)                 # (B, S, E)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)               # (B, S, V)
        return logits, hidden

def collate_batch(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys

def ids_from_tokens(tokens, stoi):
    return [stoi.get(t, stoi["<unk>"]) for t in tokens]

def train():
    tokens = read_corpus(CORPUS_PATH)
    vocab, stoi = build_vocab(tokens, MIN_FREQ)
    ids = ids_from_tokens(tokens, stoi)
    dataset = SeqDataset(ids, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    model = RNNLM(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore pad

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_tokens = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(xb)  # (B, S, V)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * xb.size(0) * xb.size(1)
            n_tokens += xb.size(0) * xb.size(1)

        ppl = math.exp(total_loss / max(1, n_tokens))
        print(f"Epoch {epoch}/{EPOCHS}  ppl={ppl:.2f}")

    # save
    torch.save({"model_state": model.state_dict(), "vocab": vocab, "stoi": stoi}, MODEL_OUT)
    print("Saved", MODEL_OUT)

def generate(prompt, max_len=50, temperature=1.0):
    ckpt = torch.load(MODEL_OUT, map_location=DEVICE)
    vocab = ckpt["vocab"]
    stoi = ckpt["stoi"]
    model = RNNLM(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokens = prompt.split()
    ids = ids_from_tokens(tokens, stoi)
    hidden = None
    generated = tokens.copy()
    inp = torch.tensor([ids[-SEQ_LEN:]], dtype=torch.long).to(DEVICE)

    for _ in range(max_len):
        logits, hidden = model(inp, hidden)
        logits = logits[0, -1] / max(1e-8, temperature)
        probs = torch.softmax(logits, dim=-1).cpu()
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_tok = vocab[next_id]
        generated.append(next_tok)
        inp = torch.tensor([[next_id]], dtype=torch.long).to(DEVICE)

    return " ".join(generated)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", type=str, default=None)
    args = parser.parse_args()
    if args.train:
        train()
    elif args.generate is not None:
        print(generate(args.generate))
    else:
        print("Use --train to train or --generate \"prompt\" to sample.")