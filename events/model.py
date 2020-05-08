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


class NodeLoss(nn.Module):

    def __init__(self, normalize_types=False):
        super().__init__()
        self.normalize_types = normalize_types
        self.log_softmax = nn.LogSoftmax()

    def forward(self, logits, targets):
        probs = self.log_softmax(logits)

        if self.normalize_types: # renormalize for types (not occurrences)
            targets = (targets > 0).long()
            targets = targets / targets.sum(dim=0)

        loss = -1 * torch.mean(targets * probs, dim=0).sum()

        return loss


class IdentityEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, g_batch, g, x):
        graph_embs = [torch.mean(x[g.entities], dim=0)] # don't use GNN for empty graph
        for i in range(g_batch.num_graphs):
            graph_embs.append(torch.mean(g_batch.x[g_batch.batch == i], dim=0))
        graph_embs = torch.stack(graph_embs)
        return graph_embs



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

        for i in range(g_batch.num_graphs):
            graph_embs.append(torch.mean(x[g_batch.batch == i], dim=0))
        graph_embs = torch.stack(graph_embs)
        return graph_embs


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
        self.graph_embedder = GCNConvEmbedder(in_channels=768, out_channels=100)
        # self.graph_embedder = IdentityEmbedder()
        # self.graph_embedder = TransformerEmbedder()

        self.event_node_classifier = nn.Linear(100, len(node_type_dictionary))
        self.node_type_dictionary = node_type_dictionary
        self.id_to_node_type = {v: k for k, v in self.node_type_dictionary.items()}
        self.node_loss = NodeLoss()

    def forward(self, batch):
        encoding = batch['encoding']
        input_ids = torch.tensor(encoding.ids).unsqueeze(0).cuda()[:, :512]
        attention_mask = torch.tensor(encoding.attention_mask).unsqueeze(0).cuda()[:, :512]
        token_emb = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask)[0].squeeze(0)
        g = batch['graph']
        node_text_emb = []
        for start, end in g.node_spans:
            if start >= 512:
                start = min(len(encoding.tokens)-1, 511)
                end = min(len(encoding.tokens), 512)
            emb = torch.mean(token_emb[start:end], dim=0)
            node_text_emb.append(emb)

        node_text_emb = torch.stack(node_text_emb) # FIXME train/test mismatch, I don't have textspans for event triggers
        node_type_emb = self.node_type_emb(torch.tensor(g.node_type).cuda())

        x = node_text_emb + node_type_emb

        partial_graphs = []
        for i, v in enumerate(batch["order"]):
            subset = g.entities + batch["order"][:i]
            edge_index, edge_attr = subgraph(subset=subset,
                                             edge_index=g.edge_index.cuda(),
                                             edge_attr=g.edge_attr.cuda())
            partial_graphs.append(Data(x=x[subset], edge_index=edge_index, edge_attr=edge_attr))
        subset = list(range(g.num_nodes))
        edge_index, edge_attr = subgraph(subset=subset,
                                         edge_index=g.edge_index.cuda(),
                                         edge_attr=g.edge_attr.cuda())
        partial_graphs.append(Data(x=x[subset], edge_index=edge_index, edge_attr=edge_attr))

        batch = Batch.from_data_list(partial_graphs)

        graph_embs = self.graph_embedder(batch, g, x)
        node_logits = self.event_node_classifier(graph_embs)

        return {"node_logits": node_logits}

    def forward_loss(self, batch):
        out = self(batch)
        node_logits = out["node_logits"]
        targets = batch["node_targets"].cuda()
        node_loss = self.node_loss(node_logits, targets)

        return node_loss

    def predict(self, batch):
        out = self(batch)

        return {"aux": out, "a2": []}





