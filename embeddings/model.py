import pytorch_lightning as pl
import torch
from torch import nn


class SimpleClassifier(nn.Module):
    def __init__(self, n_rel_types, n_mod_types, embedding_size=768 * 2):
        super().__init__()
        self.rel_output = nn.Linear(embedding_size, n_rel_types)
        self.mod_output = nn.Linear(embedding_size, n_mod_types)

    def forward(self, x):
        return self.rel_output(x), self.mod_output(x)

