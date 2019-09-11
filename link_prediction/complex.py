import torch
import torch.nn as nn
import math


class Complex(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(Complex, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.ent1_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.ent2_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel1_embs = nn.Embedding(num_rel, emb_dim).to(device)
        self.rel2_embs = nn.Embedding(num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent1_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent2_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel1_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel2_embs.weight.data, -sqrt_size, sqrt_size)

    def l2_loss(self):
        return ((torch.norm(self.ent1_embs.weight, p=2) ** 2) + (torch.norm(self.ent2_embs.weight, p=2) ** 2) + (
                torch.norm(self.rel1_embs.weight, p=2) ** 2) + (torch.norm(self.rel2_embs.weight, p=2) ** 2)) / 2

    def forward(self, heads, rels, tails):
        h1 = self.ent1_embs(heads)
        h2 = self.ent2_embs(heads)
        t1 = self.ent1_embs(tails)
        t2 = self.ent2_embs(tails)
        r1 = self.rel1_embs(rels)
        r2 = self.rel2_embs(rels)

        term1 = (h1 * t1) @ r1.t()
        term2 = (h2 * t2) @ r1.t()
        term3 = (h1 * t2) @ r2.t()
        term4 = (h2 * t1) @ r2.t()

        return term1 + term2 + term3 - term4
