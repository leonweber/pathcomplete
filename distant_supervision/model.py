import transformers
from torch import nn
import torch
from torch.nn import DataParallel



class BagEmbedder(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.bert = DataParallel(transformers.BertModel.from_pretrained(bert))
        self.ff_attention = nn.Linear(768, 1)

    def forward(self, token_ids, attention_masks, **kwargs):
        _, x = self.bert(token_ids, attention_masks)
        alphas = torch.sigmoid(self.ff_attention(x))
        x = torch.sum(alphas * x, dim=0)

        return x

class BagOnly(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.bag_embedder = BagEmbedder(bert, args)
        self.ff_output = nn.Linear(768, args.n_classes)
        self.no_mentions_emb = nn.Parameter(torch.zeros(768).uniform_(-0.02, 0.02))

    def forward(self, token_ids, attention_masks, entity_pos, labels, has_mentions, **kwargs):

        if has_mentions.sum() > 0:
            x = self.bag_embedder(token_ids, attention_masks)
        else:
            x = self.no_mentions_emb
        x = self.ff_output(x)

        return x

class Simple(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.tensor_emb_size = args.tensor_emb_size
        self.s_embedding = nn.Embedding(args.n_entities, self.tensor_emb_size)
        self.o_embedding = nn.Embedding(args.n_entities, self.tensor_emb_size)
        self.r_embedding = nn.Embedding(args.n_relations, self.tensor_emb_size)
        self.r_inv_embedding = nn.Embedding(args.n_relations, self.tensor_emb_size)
        self.no_bag_emb = nn.Parameter(torch.zeros(768).float().uniform_(-0.02, 0.02))
        self.ff_gate = nn.Sequential(
            nn.Linear(2 * self.tensor_emb_size, 1),
            nn.Sigmoid()
        )
        self.bag_downprojection = nn.Sequential(
            nn.Linear(768, self.tensor_emb_size),
            nn.ReLU()
        )

    def forward(self, entities, bag_emb1, bag_emb2, no_mentions_mask):
        has_mentions_mask = ~no_mentions_mask
        e1_s = self.s_embedding(entities[:, 0])
        e1_o = self.o_embedding(entities[:, 0])
        e2_s = self.s_embedding(entities[:, 1])
        e2_o = self.o_embedding(entities[:, 1])
        r = self.r_embedding.weight
        r_inv = self.r_inv_embedding.weight

        es1 = e1_s * e2_o
        es2 = e2_s * e1_o

        bag_emb1 = bag_emb1 * has_mentions_mask.float().unsqueeze(1) + self.no_bag_emb.repeat(bag_emb1.size(0), 1) * no_mentions_mask.float().unsqueeze(1)
        bag_emb2 = bag_emb2 * has_mentions_mask.float().unsqueeze(1) + self.no_bag_emb.repeat(bag_emb1.size(0), 1) * no_mentions_mask.float().unsqueeze(1)

        bag_emb1 = self.bag_downprojection(bag_emb1)
        bag_emb2 = self.bag_downprojection(bag_emb2)

        gate1 = self.ff_gate(torch.cat([es1, bag_emb1], dim=1))
        # masked_gate1 = torch.zeros_like(gate1)
        # masked_gate1[has_mentions_mask] = gate1[has_mentions_mask]

        gate2 = self.ff_gate(torch.cat([es2, bag_emb2], dim=1))
        # masked_gate2 = torch.zeros_like(gate2)
        # masked_gate2[has_mentions_mask] = gate2[has_mentions_mask]

        h1 = gate1 * bag_emb1 + (1-gate1) * es1
        h2 = gate2 * bag_emb2 + (1-gate2) * es2

        scores1 = h1 @ r.t()
        scores2 = h2 @ r_inv.t()

        return (scores1 + scores2) / 2

