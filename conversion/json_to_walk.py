import argparse
import json
from collections import defaultdict

from pathlib import Path



def get_nodes_by_pathway(pathway_lines):
    next(pathway_lines)
    nodes_by_pathway = defaultdict(set)
    for line in pathway_lines:
        fields = line.split('\t')
        try:
            nodes_by_pathway[fields[5]].add(fields[0])
            nodes_by_pathway[fields[5]].add(fields[2])
        except IndexError:
            continue

    return nodes_by_pathway


def get_walks(receptors, dna_binding_proteins, nodes_by_pathway):
    walks = defaultdict(dict)
    for pathway, nodes in nodes_by_pathway.items():
        sources = receptors & nodes
        targets = dna_binding_proteins & nodes
        allowed = nodes
        walks[pathway]['sources'] = sources
        walks[pathway]['targets'] = targets
        walks[pathway]['allowed'] = allowed

    return walks





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receptors', type=Path, required=True)
    parser.add_argument('--dna_binding', type=Path, required=True)
    parser.add_argument('--pathways', type=Path, required=True)

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

    n_walks = 0
    for walk in walks.values():
       n_walks += len(walk['sources']) * len(walk['targets'])
    print(n_walks)
