import argparse
import json

from pathlib import Path
import networkx as nx


def graph_lines_to_nx(graph_lines, edge_types=None):
    G = nx.DiGraph()
    for line in graph_lines:
        e1, r, e2 = line.split('\t')[:3]
        if edge_types and r not in edge_types:
            continue

        G.add_edge(e1, e2)

    return G


def get_shortest_path_nodes(walk, G: nx.DiGraph):
    nodes = set()
    for source in walk['sources']:
        for target in walk['targets']:
            try:
                nodes.update(nx.shortest_path(G, source=source, target=target))
            except (nx.exception.NetworkXNoPath, nx.exception.NodeNotFound):
                continue

    return list(nodes)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--walks', required=True, type=Path)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--edge_types', default=None)

    args = parser.parse_args()

    if args.edge_types:
        args.edge_types = args.edge_types.split(',')

    result = {}
    with args.walks.open() as f:
        walks = json.load(f)

    with args.graph.open() as f:
        G = graph_lines_to_nx(f, edge_types=args.edge_types)

    for pw_name, walk in walks.items():
        result[pw_name] = get_shortest_path_nodes(walk, G)

    with args.output.open('w') as f:
        json.dump(result, f)
