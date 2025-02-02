import torch

vocab_size = 25000
batch_size = 64
block_size = 64
max_iters = 1000
eval_interval = 1
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 64
n_head = 4
mlp_layer = 4
n_modules = 4
dropout = 0.0