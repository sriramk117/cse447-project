import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import random
import os
import gc

from hyperparameters import *

data_path = "cse447-project/datasets/shakespeare.txt"
checkpoint_path = "model_checkpoint.pt"
input_file = "cse447-project/evaluation/input.txt"
output_file = "cse447-project/evaluation/output.txt"

encoding = 'utf-8'

def encode(s):
    res = [] 
    for line in tqdm(s, "Encoding"):
        if "\n" in line:
            line = line.rstrip("\n")
        char_encode = []
        for c in line:
            if ord(c) < vocab_size:
                char_encode.append(ord(c))
        res.append(torch.tensor(char_encode, dtype=torch.long))
    return res

def decode(tensor: torch.Tensor) -> str:
    return ''.join(chr(c) for c in tensor.item())

with open(data_path, 'r', encoding=encoding) as f:
    text = f.readlines()
data = encode(text)
del text
gc.collect()

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    x = []
    y = []
    for line in tqdm(d, f"Getting {split} batch"):
        if len(line) <= block_size + 1:
            continue
        idx = random.randint(0, len(line) - block_size - 1)
        x.append(line[idx : idx+block_size])
        y.append(line[idx+1 : idx+block_size+1])
    x_tensor = torch.stack(x)
    y_tensor = torch.stack(y)
    del x, y
    gc.collect()
    return x_tensor, y_tensor

class Transformer_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.position_embeddings = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=mlp_layer_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            device=device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_modules,
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
def evaluate(model):
    out = {}
    model.eval()

    '''
    dataloader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size)
    losses = []
    for xb, yb in tqdm(dataloader, "Loss estimation (Train)"):
        _, loss = model(xb, yb)
        losses.append(loss.item())
    out['train'] = sum(losses) / len(losses)
    '''
    x_v, y_v = get_batch('val')
    print(len(y_v))
    dataloader = DataLoader(TensorDataset(x_v, y_v), batch_size=batch_size, shuffle=True)
    losses = []
    preds = []
    for xb, yb in tqdm(dataloader, "Loss estimation (Val)"):
        _, loss = model(xb.to(device), yb.to(device))
        top_inds, _ = predict_next(model, xb.to(device))
        # top 3 tokens
        top_tokens = []
        for i in range(3):
            token_id = top_inds[0, i].item()
            top_tokens.append(chr(token_id))
        preds.append(top_tokens)
        losses.append(loss.item())
    out['val_loss'] = sum(losses) / len(losses)
    
    # Calculate accuracy
    sum_acc = 0
    
    for i in range(len(preds)):
        print(preds[i], chr(y_v[i, -1].item()))
        for j in range(3):
            if preds[i][j] == chr(y_v[i, -1].item()):
                sum_acc += 1
                break
    out['val_acc'] = sum_acc / len(preds)

    model.train()

    return out

def train():
    model = Transformer_Model().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    checkpoint_path = "cse447-project/checkpoints/model_checkpoint.pt"
    
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
        x_t, y_t = get_batch('train')
        train_dataloader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)

        for x_batch, y_batch in tqdm(train_dataloader, desc="Epoch Progress"):
            logits, loss = model(x_batch.to(device), y_batch.to(device))

            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            metrics = evaluate(model)
            eval_loss = metrics['val_loss']
            acc = metrics['val_acc']
            print(f"Step {step}: train loss {loss:.4f}, val loss {eval_loss:.4f}, val acc {acc:.4f}")

            checkpoint_path = f"C:/Users/st3by/Documents/CSE447/cse447-project/checkpoints/large_model_checkpoint_{step}.pt"

            log_path = "cse447-project/loss_log.txt"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Epoch {step}: Train Loss: {loss:.4f}, Val Loss: {eval_loss:.4f}\n")

            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': model.optimizer.state_dict(),
                'step': step
            }, checkpoint_path)
    
    print("Training Complete")

def predict():
    def checkpoint_path(epoch):
        return f"model_checkpoint{epoch}.pt"

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
    #predict()
    train()

if __name__ == "__main__":
    main()