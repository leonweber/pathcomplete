from torch import nn
import torch


class BagOnly(nn.Module):
    def __init__(self, tensor_emb_size, bag_emb_size, n_entities, n_relations):
        super().__init__()
        self.output_layer1 = nn.Linear(bag_emb_size, n_relations)
        self.output_layer2 = nn.Linear(bag_emb_size, n_relations)

    def forward(self, entities, bag_emb1, bag_emb2, no_mentions_mask):
        logits1 = self.output_layer1(bag_emb1)
        logits2 = self.output_layer2(bag_emb2)
        return (logits1 + logits2)/2, None, None


class Simple(nn.Module):
    def __init__(self, tensor_emb_size, bag_emb_size, n_entities, n_relations):
        super().__init__()
        self.tensor_emb_size = tensor_emb_size
        self.bag_emb_size = bag_emb_size
        self.s_embedding = nn.Embedding(n_entities, tensor_emb_size)
        self.o_embedding = nn.Embedding(n_entities, tensor_emb_size)
        self.r_embedding = nn.Embedding(n_relations, tensor_emb_size)
        self.r_inv_embedding = nn.Embedding(n_relations, tensor_emb_size)
        self.ff_gate = nn.Sequential(
            nn.Linear(2 * tensor_emb_size, 1),
            nn.Sigmoid()
        )
        self.bag_downprojection = nn.Sequential(
            nn.Linear(bag_emb_size, tensor_emb_size),
            nn.ReLU()
        )

    def forward(self, entities, bag_emb1, bag_emb2, no_mentions_mask):
        has_mentions_mask = ~no_mentions_mask
        e1_s = self.s_embedding(entities[:, 0])
        e1_o = self.o_embedding(entities[:, 0])
        e2_s = self.s_embedding(entities[:, 1])
        e2_o = self.o_embedding(entities[:, 1])
        r = self.r_embedding.weight
        r_inv = self.r_embedding.weight

        es1 = e1_s * e2_o
        es2 = e2_s * e1_o

        bag_emb1 = self.bag_downprojection(bag_emb1)
        bag_emb2 = self.bag_downprojection(bag_emb2)

        gate1 = self.ff_gate(torch.cat([es1, bag_emb1], dim=1))
        masked_gate1 = torch.zeros_like(gate1)
        masked_gate1[has_mentions_mask] = gate1[has_mentions_mask]

        gate2 = self.ff_gate(torch.cat([es2, bag_emb2], dim=1))
        masked_gate2 = torch.zeros_like(gate2)
        masked_gate2[has_mentions_mask] = gate2[has_mentions_mask]

        h1 = masked_gate1 * bag_emb1 + (1-masked_gate1) * es1
        h2 = masked_gate2 * bag_emb2 + (1-masked_gate2) * es2

        scores1 = h1 @ r.t()
        scores2 = h2 @ r_inv.t()

        return (scores1 + scores2) / 2, masked_gate1, masked_gate2

