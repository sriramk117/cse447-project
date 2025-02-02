import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os

vocab_size = 256
batch_size = 64
block_size = 32
max_iters = 1000000
eval_interval = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

checkpoint_path = "model_checkpoint.pt"
input_file = "cse447-project/evaluation/input.txt"
output_file = "cse447-project/evaluation/output.txt"

encoding = 'utf-8'

def encode(s: str) -> torch.tensor:
    res = []
    for c in s:
        if ord(c) < vocab_size:
            res.append(ord(c))
    return torch.tensor(res, dtype=torch.long)

def decode(tensor: torch.Tensor) -> str:
    return ''.join(chr(c) for c in tensor.item())

with open('cse447-project/datasets/shakespeare.txt', 'r', encoding=encoding) as f:
    text = f.read()

data = encode(text)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x  = torch.stack([d[i : i+block_size]   for i in ix])
    y  = torch.stack([d[i+1 : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Transformer_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_layer,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.loss_fn = F.cross_entropy

    def forward(self, x, target=None):
        # Assume x.shape = (B, T) and target.shape = (B, T)
        B, T = x.shape

        if T > block_size:
            x = x[:, :block_size]
            T = block_size
            if target is not None:
                target = target[:, :block_size]

        # Converts to token and pos embeddings, (B, T, n_embd), each sample has a (T, n_embd) matrix
        token_embed = self.token_embeddings(x)
        pos = torch.arange(0, T, device=x.device)
        pos_embed = self.position_embeddings(pos) #(T, n_embd)
        pos_embed.unsqueeze(0) # Unsqueeze to (1, T, n_embd), since position embed is the same for all batches
        x = token_embed + pos_embed
        
        mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
            ).bool()

        x = self.transformer_encoder(x, mask=mask)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if target is not None:
            B, T, V = logits.shape
            logits_ = logits.view(B*T, V)
            targets_ = target.view(B*T)
            loss = self.loss_fn(logits_, targets_)
        return logits, loss

@torch.no_grad()
def predict_next(model, idx):
    model.eval()
    logits, _ = model(idx)
    logits_last = logits[:, -1, :]
    probs = F.softmax(logits_last, dim=-1)
    top_values, top_indices = torch.topk(probs, k=3, dim=-1)
    return top_indices, top_values

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

def train():
    model = Transformer_Model().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    start_step = 0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step'] + 1
        print(f"Checkpoint loaded at Step {start_step}")
    else:
        print("No checkpoint")

    model.train()
    for step in range(start_step, max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': model.optimizer.state_dict(),
                'step': step
            }, checkpoint_path)

        x_batch, y_batch = get_batch('train')
        logits, loss = model(x_batch, y_batch)

        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.optimizer.step()
    
    print("Training Complete")

def predict():
    checkpoint_path = "model_checkpoint.pt"

    model = Transformer_Model().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step'] + 1
        print(f"Checkpoint loaded at Step {start_step}")
    else:
        print("No checkpoint found. Using untrained model.")

    model.eval()
    with open(input_file, "r", encoding=encoding) as fin, \
         open(output_file, "w", encoding=encoding) as fout:
        
        for line in fin:
            line_str = line.strip()
            if not line_str:
                fout.write("\n")
                continue

            line_ids = encode(line_str).to(device)
            line_tensor = line_ids.unsqueeze(0)

            top_inds, top_vals = predict_next(model, line_tensor)

            top3_tokens = []
            for i in range(3):
                token_id = top_inds[0, i].item()
                top3_tokens.append(chr(token_id))
            top3_str = ''.join(top3_tokens)
            fout.write(line_str + "     " + top3_str + "\n")

def main():
    predict()
    #train()

if __name__ == "__main__":
    main()