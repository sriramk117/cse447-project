import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import random

from hyperparameters import *
from config import *

class Transformer_Model(nn.Module):
    def __init__(self, device="cpu"):
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
        B, T = x.shape

        if T > block_size:
            x = x[:, :block_size]
            T = block_size
            if target is not None:
                target = target[:, :block_size]

        token_embed = self.token_embeddings(x)
        pos = torch.arange(0, T, device=x.device)
        pos_embed = self.position_embeddings(pos)
        pos_embed.unsqueeze(0)
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
    
class TextDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        encoded = encode_line(line)
        if len(encoded) <= block_size + 1:
            return None, None

        start_idx = random.randint(0, len(encoded) - block_size - 1)
        x_chunk = encoded[start_idx : start_idx + block_size]
        y_chunk = encoded[start_idx + 1 : start_idx + block_size + 1]

        x_tensor = torch.tensor(x_chunk, dtype=torch.long)
        y_tensor = torch.tensor(y_chunk, dtype=torch.long)
        return x_tensor, y_tensor
    
def collate_fn(data):
    data = [sample for sample in data if sample[0] is not None]
    x, y = zip(*data)
    x_tensor = torch.stack(x)
    y_tensor = torch.stack(y)
    return x_tensor, y_tensor

def load_dataset(file_path, train_ratio=0.9):
    with open(file_path, 'r', encoding=encoding) as f:
        texts = f.readlines()
    
    split = int(train_ratio*len(texts))
    return texts[:split], texts[split:]

def encode_line(line):
    return [ord(c) for c in line if ord(c) < vocab_size]

@torch.no_grad()
def predict_next(model, seq):
    model.eval()
    logits, _ = model(seq)
    logits_last = logits[:, -1, :]
    probs = F.softmax(logits_last, dim=-1)
    top_values, top_indices = torch.topk(probs, k=3, dim=-1)
    return top_indices, top_values

@torch.no_grad()
def evaluate(model, eval, device="cpu"):
    out = {}
    model.eval()
    dataloader = DataLoader(TextDataset(eval), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    losses = []
    batch_acc = []

    for x_batch, y_batch in tqdm(dataloader, "Eval Loss"):
        _, loss = model(x_batch.to(device), y_batch.to(device))
        losses.append(loss.item())
        top_inds, _ = predict_next(model, x_batch.to(device))

        batch_correct = (y_batch[:, -1].view(-1, 1).to('cpu') == top_inds.to('cpu')).any(dim=1)
        batch_acc.append(sum(batch_correct.float()))
    
    out['val_loss'] = sum(losses) / len(losses)
    out['val_acc'] = (sum(batch_acc)/len(eval)).item()
    return out

def train():
    train, eval = load_dataset(file_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer_Model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    start_epoch = 0
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded at epoch {start_epoch}.")
    else:
        print("No checkpoint found.")

    model.train()

    for epoch in range(start_epoch, num_epochs):
        train_dataloader = DataLoader(TextDataset(train), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

        for x_batch, y_batch in tqdm(train_dataloader, desc="Epoch Progress"):
            _, loss = model(x_batch.to(device), y_batch.to(device))

            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()

        if epoch % eval_interval == 0:
            metrics = evaluate(model, eval, device)
            eval_loss = metrics['val_loss']
            acc = metrics['val_acc']
            print(f"Epoch {epoch}: Train Loss {loss:.4f}, Val Loss {eval_loss:.4f}, Val Acc {acc:.4f}")

            checkpoint_file_path = checkpoint_path + f"large_model_checkpoint_{epoch}.pt"
            with open(loss_log_path, "a", encoding="utf-8") as f:
                f.write(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {eval_loss:.4f}\n")

            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': model.optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_file_path)
    
    print("Training Complete")

def main():
    train()

if __name__ == "__main__":
    main()