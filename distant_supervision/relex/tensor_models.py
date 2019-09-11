from torch import nn
import torch


class Complex(nn.Module):
    def __init__(self, rank, n_entities, n_relations):
        super().__init__()
        self.rank = rank
        self.e1 = nn.Embedding(n_entities, rank)
        self.e2 = nn.Embedding(n_entities, rank)
        self.r1 = nn.Embedding(n_relations, rank)
        self.r2 = nn.Embedding(n_relations, rank)

    def forward(self, entities):
        s1 = self.e1(entities[:, 0])
        s2 = self.e2(entities[:, 0])
        o1 = self.e1(entities[:, 1])
        o2 = self.e2(entities[:, 1])
        r1 = self.r1.weight
        r2 = self.r2.weight

        term1 = (s1 * o1) @ r1.t()
        term2 = (s2 * o2) @ r1.t()
        term3 = (s1 * o2) @ r2.t()
        term4 = (s2 * o1) @ r2.t()

        return term1 + term2 + term3 - term4, torch.cat([s1, s2, o1, o2], dim=1)

    @property
    def summary_size(self):
        return self.rank * 4



