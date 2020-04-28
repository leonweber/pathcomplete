from pathlib import Path
from typing import List, Dict

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import subgraph
from transformers import BertModel
import torch_geometric



class EventExtractor(nn.Module):

    def __init__(self, trial: optuna.Trial, bert: Path, hidden_dim,
                 node_type_dictionary, edge_type_dictionary,
                 batch_norm=False, residual=True,
                 n_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(bert))

        self._embeddings = {}


    def forward(self, batch):
        token_emb = self.bert(input_ids=torch.tensor(batch['input_ids']).unsqueeze(0),
                              attention_mask=batch['attention_mask'].unsqueeze(0))[0].squeeze(0)

    def forward_loss(self, batch):
        encoding = batch['encoding']
        input_ids = torch.tensor(encoding.ids).unsqueeze(0).cuda()
        attention_mask = torch.tensor(encoding.attention_mask).unsqueeze(0).cuda()
        token_emb = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask)[0].squeeze(0)
        partial_graphs = []
        g = batch['graph']
        for i, _ in enumerate(g.topological_order):
            subset = g.entities + g.topological_order[:i]
            partial_graphs.append(subgraph(subset=subset,
                                           edge_index=g.edge_index,
                                           edge_attr=g.edge_attr))

        pass
