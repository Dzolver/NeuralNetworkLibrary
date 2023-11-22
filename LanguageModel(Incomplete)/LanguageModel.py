import torch
import torch.nn as nn
import numpy as np
import os
from torch.nn.utils import clip_grad_norm_


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000
batch_size = 20
seq_length =30
learning_rate = 0.002

#Let's load the data (Penn Treebank dataset)

