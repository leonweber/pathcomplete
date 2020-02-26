import json
import logging
import math
from collections import defaultdict

import h5py
import networkx as nx
import numpy as np
import dgl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging


class SuperPathDataset(Dataset):
    def __init__(self, interactome, pathways,
                 interactome_is_undirected=True,
                 interactome_texts=None):
        g = nx.readwrite.read_adjlist(interactome, create_using=nx.DiGraph())
        if interactome_is_undirected:
            g.add_edges_from(g.reverse().edges)
        self.nodes = sorted(g.nodes)
        self.interactome_edges = sorted(f"{u},{v}" for u, v in g.edges)
        self.interactome = dgl.DGLGraph()
        self.interactome.from_networkx(g)
        self.interactome.add_edges(self.interactome.nodes(), self.interactome.nodes())

        if interactome_texts:
            embs = []
            f = h5py.File(interactome_texts, mode='r', driver='core')
            for edge in tqdm(self.interactome_edges, desc='Loading text embeddings'):
                if edge in f['embeddings']:
                    emb = f['embeddings'][edge]
                else:
                    emb = np.zeros_like(embs[-1])
                embs.append(emb)
        else:
            self.interactome_texts = None

        self.n_nodes = len(self.nodes)
        self._node_ids = torch.arange(self.n_nodes).long()
        self.sigma = 1
        self.keep_prob = 0.5

        with pathways.open() as f:
            self.pathways = {k: set(v) for k,v in json.load(f).items()}

        self.pathway_ids = sorted(self.pathways)

    def __getitem__(self, item):
        pw_id = self.pathway_ids[item]
        true_pw_nodes = self.pathways[pw_id]
        n_sample = int(np.round(np.random.normal(len(true_pw_nodes)//2, np.sqrt(len(true_pw_nodes)//2))))

        # n_sample should be in [2, len(true_pw_nodes)-1] so that something can be predicted
        n_sample = max(n_sample, 2)
        n_sample = min(n_sample, len(true_pw_nodes)-1)

        indicated_pw_nodes = np.random.choice(sorted(true_pw_nodes), size=n_sample, replace=False)

        labels = torch.tensor([node in true_pw_nodes for node in self.nodes]).float()
        pw_indicators = torch.tensor([node in indicated_pw_nodes for node in self.nodes]).float()


        return {
            'node_ids': self._node_ids,
            'pw_indicators': pw_indicators,
            'labels': labels
        }

    def __len__(self):
        return len(self.pathway_ids)










