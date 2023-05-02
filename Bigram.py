import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many sequences processed in parallel
block_size = 256  # context length for predictions
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# the unique characters from the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping for chars:int & int:chars
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode/decode fn's for input/output
encode = lambda s: [stoi[c] for c in s]  # string -> int
decode = lambda l: "".join([itos[i] for i in l])  # int -> string

# store encoded dataset into a torch.tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split into training and validation sets - 90/10
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]


def get_batch(split):
    # generate a set of inputs `x` and targets `y`
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    ## Single headed self-attention

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, T, 16) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # normalize (B, T, T)
        v = self.value(x)  # (B, T, C)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    # Multiple heads of self-attention in parallel

    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # concat over the channel dim
        out = self.proj(out)
        # projection back into the linear pathway
        return out


class FeedForward(nn.Module):
    # simple linear layer followed by a non-linearity

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # Transformer block: communication follow by computation

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: # of heads we want
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = token_emb + pos_emb
        x = self.blocks(x)  # apply one head of self-attention (B,T,C)
        logits = self.lm_head(token_emb)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # cross_entropy requires Channels to be 2nd element in tensor, reshape here.
            # targets needs to be a 1 dim tensor - reshape here too.
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # this function is somewhat generic for other model designs
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predicitons
            logits, loss = self(idx_cond)
            # focus on the last time step - last element in the time dim
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # get loss on train and val sets periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # zero out gradients from previous step
    loss.backward()  # getting grads for all parameters
    optimizer.step()  # using grads to train

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)

# creates a [1,1] tensor, with 0 as placeholder. Generates 100 new tokens -> converts to python list
# decodes list of int's to chars and prints our jibberish.
print(
    decode(
        m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[
            0
        ].tolist()
    )
)
