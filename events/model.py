from pathlib import Path
from typing import List, Dict

import numpy as np
import networkx as nx
import optuna
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph
from transformers import BertModel
import torch_geometric

from events.dataset import MAX_LEN


class NodeLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, logits_by_step, targets_by_step):
        loss = 0
        for logits, targets in zip(logits_by_step, targets_by_step):
            probs = self.log_softmax(logits)
            loss += -1 * torch.mean(targets * probs, dim=0).sum()

        return loss/len(logits_by_step)



class GCNConvEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.ModuleList([SAGEConv(self.in_channels, self.out_channels)])
        for _ in range(n_layers-1):
            self.layers.append(SAGEConv(self.out_channels, self.out_channels))

    def forward(self, g_batch, g, x):
        graph_embs = [] # don't use GNN for empty graph
        x = g_batch.x
        for layer in self.layers:
            x = layer(x, g_batch.edge_index)

        return x


class TransformerEmbedder(nn.Module):
    def __init__(self, n_edge_types):
        super().__init__()
        self.edge_type_embedding = nn.Embedding(n_edge_types, 768)
        layer = nn.TransformerEncoderLayer(d_model=100, nhead=4, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=3)

    def forward(self, g_batch, g, x):
        pass


class EventExtractor(nn.Module):

    def __init__(self, trial: optuna.Trial, bert: Path, hidden_dim,
                 node_type_dictionary, edge_type_dictionary,
                 batch_norm=False, residual=True,
                 n_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(bert))
        self.node_type_emb = nn.Embedding(len(node_type_dictionary), 768)
        self.node_marker_emb = nn.Embedding(2, 768)
        self.graph_embedder = GCNConvEmbedder(in_channels=768, out_channels=100)
        self.dummy_node = nn.Parameter(torch.zeros(100).normal_(0, 0.02))
        # self.graph_embedder = IdentityEmbedder()
        # self.graph_embedder = TransformerEmbedder()

        self.node_classifier = nn.Linear(100, 1)
        self.node_type_dictionary = node_type_dictionary
        self.id_to_node_type = {v: k for k, v in self.node_type_dictionary.items()}
        self.node_loss = NodeLoss()

    def forward(self, batch):
        encoding = batch['encoding']
        input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0).cuda()[:, :MAX_LEN]
        attention_mask = torch.tensor(encoding["attention_mask"]).unsqueeze(0).cuda()[:, :MAX_LEN]
        token_emb = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask)[0].squeeze(0)
        g = batch['graph']
        node_text_emb = []
        for start, end in g.node_spans:
            if start >= MAX_LEN:
                start = min(len(encoding["offset_mapping"])-1, MAX_LEN-1)
                end = min(len(encoding["offset_mapping"]), MAX_LEN)
            emb = torch.mean(token_emb[start:end], dim=0)
            node_text_emb.append(emb)

        node_text_emb = torch.stack(node_text_emb)
        node_type_emb = self.node_type_emb(torch.tensor(g.node_type).cuda())

        x = node_text_emb + node_type_emb

        partial_graphs = []

        for i_order, v in enumerate(batch["order"]):
            subset = []
            subset += g.entities
            for span in batch["order"][:i_order]:
                for i_node, node_span in enumerate(g.node_spans):
                    if span == node_span:
                        subset.append(i_node)
            edge_index, edge_attr = subgraph(subset=subset,
                                             edge_index=g.edge_index.cuda(),
                                             edge_attr=g.edge_attr.cuda())
            partial_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, is_candidate=is_candidate))

        g_batch = Batch.from_data_list(partial_graphs, follow_batch=["is_candidate"])

        x = self.graph_embedder(g_batch, g, x)
        node_logits = self.node_classifier(x[g_batch.is_candidate])
        node_logits_by_step = []
        for i_order in range(g_batch.num_graphs):
            mask = g_batch.is_candidate_batch[g_batch.is_candidate] == i_order
            node_logits_by_step.append(node_logits[mask].squeeze(1))

        return {"node_logits": node_logits_by_step}

    def forward_loss(self, batch):
        out = self(batch)
        node_logits = out["node_logits"]
        targets = [i.cuda() for i in batch["node_targets"]]
        node_loss = self.node_loss(node_logits, targets)

        return node_loss

    def predict(self, batch):
        out = self(batch)

        return {"aux": out, "a2": []}





