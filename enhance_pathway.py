import sys
import numpy as np
from pypathway.utils import IdMapping
from collections import defaultdict
import networkx as nx


def build_graph(f):
    G = nx.Graph()
    edge_labels = {}
    for line in f:
        e1, r, e2, supports = line.strip('\n').split('\t')
        G.add_edge(e1, e2)
        edge_labels[(e1, e2)] = r
    return G, edge_labels


def get_name(id_):
        try:
            name = IdMapping.convert([id_], species='hsa', source='ENTREZID', target='SYMBOL')[0][1][0]
        except IndexError:
            name = id_

        return name


def get_node_labels(G):
    result = {}
    for node in G.nodes:
        result[node] = get_name(node)
    return result




def enhance(pathway, kb):
    reactions = defaultdict(list)
    nodes = set()
    for l in pathway:
        entry = l.strip('\n').split('\t')
        nodes.update(entry[0].split(';'))
        nodes.update(entry[2].split(';'))
    for l in kb:
        entry = l.split('\t')
        if entry[0] in nodes and entry[2] in nodes and entry[1] == 'reaction':
            try:
                e1 = IdMapping.convert([entry[0]], species='hsa', source='ENTREZID', target='SYMBOL')[0][1][0]
            except IndexError:
                e1 = entry[0]
            try:
                e2 = IdMapping.convert([entry[2]], species='hsa', source='ENTREZID', target='SYMBOL')[0][1][0]
            except IndexError:
                e2 = entry[2]
            key = tuple(sorted([e1, e2]))
            reactions[key].append(entry[3].strip())
                    
    return reactions

