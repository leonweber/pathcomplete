import argparse
import json
import os

from scipy.special import expit
from pathlib import Path
import networkx as nx
from tqdm import tqdm


def graph_lines_to_nx(graph_lines, edge_types=None, has_scores=False):
    G = nx.DiGraph()
    for line in graph_lines:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        if has_scores:
            e1, r, e2, score = fields[:4]
            score = expit(float(score))
        else:
            e1, r, e2 = fields[:3]
            score = 1.0
        if edge_types and r not in edge_types:
            continue

        G.add_edge(e1, e2, score=score)

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


def get_max_flow_nodes(walk, G: nx.DiGraph):
    nodes = set()
    sources = set(walk['sources'])
    sinks = set(walk['targets'])

    sources = sources - sinks
    sinks = sinks - sources

    if not (sources and sinks):
        return []

    for source in sources:
        G.add_edge('_SUPER_SOURCE', source)
    for sink in sinks:
        G.add_edge(sink, '_SUPER_SINK')

    flow_value, flow_dict = nx.maximum_flow(G, '_SUPER_SOURCE', '_SUPER_SINK', capacity='score')

    for u, edge_flows in flow_dict.items():
        for v, edge_flow_val in edge_flows.items():
            if edge_flow_val > 0:
                nodes.add(u)
                nodes.add(v)
    if '_SUPER_SOURCE' in nodes:
        nodes.remove('_SUPER_SOURCE')
    if '_SUPER_SINK' in nodes:
        nodes.remove('_SUPER_SINK')

    return list(nodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--walks', required=True, type=Path)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--edge_types', default=None)
    parser.add_argument('--type', default='shortest_path', choices=['shortest_path', 'max_flow'])
    parser.add_argument('--has_scores', action='store_true')

    args = parser.parse_args()

    if args.edge_types:
        args.edge_types = args.edge_types.split(',')

    result = {}
    with args.walks.open() as f:
        walks = json.load(f)

    with args.graph.open() as f:
        G = graph_lines_to_nx(f, edge_types=args.edge_types, has_scores=args.has_scores)

    if args.type == 'shortest_path':
        baseline_func = get_shortest_path_nodes
    elif args.type == 'max_flow':
        baseline_func = get_max_flow_nodes

    for pw_name, walk in tqdm(walks.items(), total=len(walks)):
        # noinspection PyUnboundLocalVariable
        result[pw_name] = baseline_func(walk, G)

    os.makedirs(args.output.parent, exist_ok=True)
    with args.output.open('w') as f:
        json.dump(result, f)
