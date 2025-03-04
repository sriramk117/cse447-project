import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.ao.quantization as quant
from tqdm import tqdm
import os
import random
import time
import argparse

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
    start_time = time.time()
    model.eval()
    logits, _ = model(seq)
    logits_last = logits[:, -1, :]
    probs = F.softmax(logits_last, dim=-1)
    top_values, top_indices = torch.topk(probs, k=3, dim=-1)
    elapsed_time = time.time() - start_time
    #print(f"Prediction Time: {elapsed_time:.4f}")
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

def predict(test_data, test_output, checkpoint_load_path, device="cpu"):
    start_time = time.time()

    model = Transformer_Model().to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(checkpoint_load_path)
    if os.path.isfile(checkpoint_load_path):
        checkpoint = torch.load(checkpoint_load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded at Epoch {start_epoch}")
    else:
        print("No checkpoint found. Using untrained model.")

    model.eval()
    with open(test_data, "r", encoding=encoding) as fin, \
         open(test_output, "w", encoding=encoding) as fout:
        
        for line in fin:
            line_str = line.strip()
            if not line_str:
                fout.write("\n")
                continue
            
            line_ids = encode_line(line_str)
            if len(line_ids) == 0:
                fout.write("aeo\n")
                continue
            line_ids = torch.tensor(line_ids).to(device)
            line_tensor = line_ids.unsqueeze(0)

            top_inds, _ = predict_next(model, line_tensor)

            top3_tokens = []
            for i in range(3):
                token_id = top_inds[0, i].item()
                top3_tokens.append(chr(token_id))
            top3_str = ''.join(top3_tokens)
            fout.write(top3_str + "\n")
    elapsed_time = time.time() - start_time
    #print(f"Elapsed Time For Entire Test Dataset: {elapsed_time:.4f}")

def main():
    #train()
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        checkpoint_load_path = "work/large_model_checkpoint_6.pt"
        predict(args.test_data, args.test_output, checkpoint_load_path)

if __name__ == "__main__":
    main()