import torch

block_size = 128
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
mlr_layer = 4
dropout = 0.0