import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = "mps" if torch.backends.mps.is_available() else "cpu"
eval_iters = 200
# ------------

torch.manual_seed(1337)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Batch creation
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# @torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Init model and optimizer
model = BigramLanguageModel(vocab_size)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Track loss
train_losses = []
val_losses = []
eval_steps = []

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eval_steps.append(iter)
        print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Plot the losses
# plt.plot(eval_steps, train_losses, label="Train Loss")
# plt.plot(eval_steps, val_losses, label="Val Loss")
# plt.xlabel("Step")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss over Time")
# plt.legend()
# plt.grid(True)
# plt.show()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
