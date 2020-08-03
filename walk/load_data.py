import itertools
import json
import logging
import math
import random
from collections import defaultdict, deque
from copy import copy
from operator import itemgetter
from pathlib import Path

import h5py
import networkx as nx
import numpy as np
import dgl
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging


def get_shortest_connecting_edges(components, G_base):
    edges = set()
    if len(components) > 1:
        component_nodes = []
        for i, component in enumerate(components):
            G_base.add_node(i)
            component_nodes.append(i)
            for node in component:
                G_base.add_edge(i, node)

        for i, j in itertools.combinations(component_nodes, 2):
            for path in nx.all_shortest_paths(G_base, source=i, target=j):
                for path_idx in range(1, len(path)-2):
                    edges.add((path[path_idx], path[path_idx+1]))
                    edges.add((path[path_idx+1], path[path_idx])) # we assume G is directed and G_base is undirected
                    pass

        for i in component_nodes:
            G_base.remove_node(i)

    return edges


class SuperPathDataset(Dataset):
    @staticmethod
    def collate(batch):
        assert len(batch) == 1
        return batch[0]


    def __init__(self, interactome, pathways,
                 interactome_texts=None):

        self.label_to_id = {}
        self.node_to_id = {}

        self.G_interactome: nx.Graph = nx.read_edgelist(interactome,
                                                        create_using=nx.Graph)
        for u in self.G_interactome.nodes:
            if u not in self.node_to_id:
                self.node_to_id[u] = len(self.node_to_id)

        self.pathways = {}
        with pathways.open() as f:
            for pw_id, pw_triples in json.load(f).items():
                G = nx.DiGraph()
                edge_attrs = defaultdict(lambda: {'labels': set()})
                for triple in pw_triples:
                    e1, r, e2 = triple.split(",")
                    if r not in self.label_to_id:
                        self.label_to_id[r] = len(self.label_to_id)
                    if e1 in self.G_interactome.nodes and e2 in self.G_interactome.nodes:
                        G.add_edge(e1, e2)
                        edge_attrs[(e1, e2)]['labels'].add(r)
                nx.set_edge_attributes(G, edge_attrs)
                self.pathways[pw_id] = G
        self.pathway_ids = sorted(self.pathways)

    def copy_with_pathways(self, pathways):
        self_copy = copy(self)
        with pathways.open() as f:
            self_copy.pathways = {k: set(v) for k, v in json.load(f).items()}

        self_copy.pathway_ids = sorted(self_copy.pathways)

        return self_copy

    @property
    def n_classes(self):
        return len(self.label_to_id)

    @property
    def n_nodes(self):
        return len(self.G_interactome.nodes)

    def __getitem__(self, item):
        pw_id = self.pathway_ids[item]
        G_pw = self.pathways[pw_id]

        n_sample = np.random.randint(1, len(G_pw.edges))
        revealed_edges = set(random.sample(list(G_pw.edges), n_sample))

        # components_in_interactome = list(nx.strongly_connected_components(G_pw.edge_subgraph(
        #     list(self.G_interactome.edges) +
        #     [(v, u) for u, v in self.G_interactome.edges] +
        #     list(revealed_edges)
        # )))

        components = list(
            nx.weakly_connected_components(G_pw.edge_subgraph(
                list(revealed_edges)
            )))

        # revealed_nodes = set()
        # for component in components_in_interactome:
        #     revealed_nodes.update(random.sample(component, 1))

        connecting_paths_edges = get_shortest_connecting_edges(
            components=components,
            G_base=self.G_interactome)

        true_edges = {
            (u, v): d for u, v, d in G_pw.edges(data=True) if
            self.G_interactome.has_edge(u, v)}

        seed_nodes = set()
        for edge in revealed_edges:
            seed_nodes.update(edge)
        node_queue = deque(seed_nodes)

        discovered_nodes = seed_nodes.copy()
        discovered_edges = revealed_edges.copy()
        examples = []

        while set(true_edges) - discovered_edges and node_queue: # if node_queue is empty before we found all edges, the remaining edges are not reachable
            G = G_pw.edge_subgraph(
                discovered_edges | connecting_paths_edges).copy()  # we add known pw edges _with_ their pw labels
            u = node_queue.popleft()
            edge_candidates = [(u, v) for v in self.G_interactome.neighbors(u)]
            examples.append(self.make_example(edge_candidates=edge_candidates, G=G,
                                              true_edges=true_edges))

            # at inference-time new_pw_edges should be the predictions of the classifier.
            # we assume that our classifier always predicts correctly during training
            new_pw_edges = [e for e in edge_candidates if
                            e in set(true_edges) | connecting_paths_edges]
            for _, v in new_pw_edges:
                if v not in discovered_nodes:
                    discovered_nodes.add(v)
                    node_queue.append(v)

            discovered_edges.update(new_pw_edges)

        return examples

    def make_example(self, edge_candidates, G, true_edges):
        G.add_edges_from(
            edge_candidates)  # we add candidate pw edges without their pw labels, because we want to predict them
        G_dgl = dgl.DGLGraph()
        G_dgl.from_networkx(G)
        node_ids = [self.node_to_id[n] for n in sorted(G.nodes())]
        dgl_node_ids = {n: i for i, n in enumerate(sorted(
            G.nodes()))}  # FIXME: This might be a problem if the order differs from the one used by dgl internally

        pw_indicators = torch.zeros((len(G_dgl.edges), len(self.label_to_id)))
        edge_ids = G_dgl.edge_ids([dgl_node_ids[e[0]] for e in G.edges],
                                  [dgl_node_ids[e[1]] for e in G.edges])
        for edge_id, (_, _, data) in zip(edge_ids, G.edges(data=True)):
            if 'labels' in data:
                for label in data['labels']:
                    pw_indicators[edge_id, self.label_to_id[label]] = 1

        labels = torch.zeros((len(G_dgl.edges), len(self.label_to_id)))
        labels[
        :] = -1  # labels_ij >= 0 if i is a candidate edge => only predict for those
        true_edge_names = sorted([e for e in true_edges if e in edge_candidates])
        edge_ids = G_dgl.edge_ids([dgl_node_ids[e[0]] for e in true_edge_names],
                                  [dgl_node_ids[e[1]] for e in true_edge_names])
        for edge_id, edge_name in zip(edge_ids, true_edge_names):
            labels[edge_id, :] = 0
            data = true_edges[edge_name]
            if 'labels' in data:
                for label in data['labels']:
                    labels[edge_id, self.label_to_id[label]] = 1

        return {'node_ids': torch.tensor(node_ids), 'G': G_dgl, 'labels': labels,
                'pw_indicators': pw_indicators}

    def __len__(self):
        return len(self.pathway_ids)

