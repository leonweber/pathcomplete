import torch
import torch.nn as nn
import math

class SimplE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(SimplE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.ent_h_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.ent_t_embs   = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs     = nn.Embedding(num_rel, emb_dim).to(device)
        self.rel_inv_embs = nn.Embedding(num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def l2_loss(self):
        return ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) + (torch.norm(self.rel_embs.weight, p=2) ** 2) + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)) / 2

    def forward(self, pairs):
        heads = pairs[:, 0]
        tails = pairs[:, 1]
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        R_embs = self.rel_embs.weight
        R_inv_embs = self.rel_inv_embs.weight

        scores1 = hh_embs * tt_embs @ R_embs.t()
        scores2 = ht_embs * th_embs @ R_inv_embs.t()
        return torch.clamp((scores1 + scores2) / 2, -20, 20)
