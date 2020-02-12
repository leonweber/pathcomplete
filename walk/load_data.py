from collections import defaultdict

import networkx as nx
import dgl

def load_base_interactome(lines):
    graphs = defaultdict(list)
    for line in lines[1:]:
        line = line.strip().split('\t')
        if not line:
            continue
        tail, head, weight, edge_type = line

        graph_type = ('protein', edge_type, 'protein')
        graphs[graph_type].append([head, tail])

    # graphs = {k: dgl.graph(v) for k,v in graphs.items()}
    graph = dgl.heterograph(graphs)


    return dgl.heterograph(graphs)
