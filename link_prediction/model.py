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

        self.ent_h_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.ent_t_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)
        self.rel_inv_embs = nn.Embedding(num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)

    def l2_loss(self):
        return ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) + (
                    torch.norm(self.rel_embs.weight, p=2) ** 2) + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)) / 2

    def forward(self, heads, rels, tails):
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        r_embs = self.rel_embs(rels)
        r_inv_embs = self.rel_inv_embs(rels)

        scores1 = torch.sum(hh_embs * r_embs * tt_embs, dim=1)
        scores2 = torch.sum(ht_embs * r_inv_embs * th_embs, dim=1)
        return torch.clamp((scores1 + scores2) / 2, -20, 20)

    def predict_relations(self, pairs):
        heads = pairs[:, 0]
        tails = pairs[:, 1]
        hh_embs = self.ent_h_embs(heads)
        ht_embs = self.ent_h_embs(tails)
        th_embs = self.ent_t_embs(heads)
        tt_embs = self.ent_t_embs(tails)
        R_embs = self.rel_embs.weight
        R_inv_embs = self.rel_inv_embs.weight

        scores1 = hh_embs * tt_embs @ R_embs.t()[1:] # first relation is 'NA', do not predict this
        scores2 = ht_embs * th_embs @ R_inv_embs.t()[1:] # first relation is 'NA', do not predict this
        return torch.clamp((scores1 + scores2) / 2, -20, 20)


class Complex(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(Complex, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.entity_re_embedding = nn.Embedding(num_ent, emb_dim).to(device)
        self.entity_im_embedding = nn.Embedding(num_ent, emb_dim).to(device)
        self.r_re_embedding = nn.Embedding(num_rel, emb_dim).to(device)
        self.r_im_embedding = nn.Embedding(num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(self.emb_dim)
        nn.init.uniform_(self.entity_re_embedding.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.entity_im_embedding.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.r_re_embedding.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.r_im_embedding.weight.data, -sqrt_size, sqrt_size)

    def l2_loss(self):
        return ((torch.norm(self.entity_re_embedding.weight, p=2) ** 2) + (torch.norm(self.entity_im_embedding.weight, p=2) ** 2) + (
                    torch.norm(self.r_re_embedding.weight, p=2) ** 2) + (torch.norm(self.r_im_embedding.weight, p=2) ** 2)) / 2

    def forward(self, heads, rels, tails):
        head_re = self.entity_re_embedding(heads)
        head_im = self.entity_im_embedding(heads)
        tail_re = self.entity_re_embedding(tails)
        tail_im = self.entity_im_embedding(tails)
        r_re = self.r_re_embedding(rels)
        r_im = self.r_im_embedding(rels)

        score = torch.sum(r_re * head_re * tail_re) + \
                torch.sum(r_re * head_im * tail_im) + \
                torch.sum(r_im * head_re * tail_im) - \
                torch.sum(r_im * head_im * tail_re)

        return score

    def predict_relations(self, pairs):
        heads = pairs[:, 0]
        tails = pairs[:, 1]

        head_re = self.entity_re_embedding(heads)
        head_im = self.entity_im_embedding(heads)

        tail_re = self.entity_re_embedding(tails)
        tail_im = self.entity_im_embedding(tails)

        R_re = self.r_re_embedding.weight[1:] # first relation is 'NA', do not predict this
        R_im = self.r_im_embedding.weight[1:] # first relation is 'NA', do not predict this

        score = (head_re * tail_re) @ R_re.t() +\
        (head_im * tail_im) @ R_re.t() +\
        (head_re * tail_im) @ R_im.t() -\
        (head_im * tail_re) @ R_im.t()

        return score
