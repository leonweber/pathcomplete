import argparse
import json
import random
from collections import defaultdict

from pathlib import Path
import numpy as np


SEED = 5005

np.random.seed(SEED)
random.seed(SEED)

def get_nodes_by_pathway(pathway_lines):
    next(pathway_lines)
    nodes_by_pathway = defaultdict(set)
    for line in pathway_lines:
        fields = line.split('\t')
        try:
            pathways = fields[5]
            for pathway in pathways.split(";"):
                nodes_by_pathway[pathway].add(fields[0])
                nodes_by_pathway[pathway].add(fields[2])
        except IndexError:
            continue

    return nodes_by_pathway


def get_walks(receptors, dna_binding_proteins, nodes_by_pathway):
    walks = defaultdict(dict)
    for pathway, nodes in nodes_by_pathway.items():
        sources = receptors & nodes
        targets = dna_binding_proteins & nodes
        allowed = nodes

        if sources and targets:
            walks[pathway]['sources'] = list(sources)
            walks[pathway]['targets'] = list(targets)
            walks[pathway]['allowed'] = list(allowed)


    return walks


def print_walk_statistics(walks):
    n_walks = 0
    for walk in walks.values():
        n_walks += len(walk['sources']) * len(walk['targets'])

    print("Num. walks", n_walks)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receptors', type=Path, required=True)
    parser.add_argument('--dna_binding', type=Path, required=True)
    parser.add_argument('--pathways', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)

    args = parser.parse_args()

    with args.receptors.open() as f:
        receptors = set(i.strip() for i in f)

    with args.dna_binding.open() as f:
        recepdna_binding_proteins = set(i.strip() for i in f)

    with args.pathways.open() as f:
        nodes_by_pathway = get_nodes_by_pathway(f)

    walks = get_walks(receptors=receptors,
                      dna_binding_proteins=recepdna_binding_proteins,
                      nodes_by_pathway=nodes_by_pathway)

    print("Total")
    print_walk_statistics(walks)

    with args.output.open('w') as f:
        json.dump(walks, f)

    walk_names = sorted(walks)
    np.random.shuffle(walk_names)
    idx1 = int(len(walk_names) * 0.6)
    idx2 = idx1 + int(len(walk_names) * 0.1)

    print("Train")
    train_walks = {i: walks[i] for i in walk_names[:idx1]}
    print_walk_statistics(train_walks)

    print("Dev")
    dev_walks = {i: walks[i] for i in walk_names[idx1:idx2]}
    print_walk_statistics(dev_walks)

    print("Test")
    test_walks = {i: walks[i] for i in walk_names[idx2:]}
    print_walk_statistics(test_walks)


    with args.output.with_suffix(args.output.suffix + '.train').open('w') as f:
        json.dump(train_walks, f)

    with args.output.with_suffix(args.output.suffix + '.dev').open('w') as f:
        json.dump(dev_walks, f)

    with args.output.with_suffix(args.output.suffix + '.test').open('w') as f:
        json.dump(test_walks, f)

