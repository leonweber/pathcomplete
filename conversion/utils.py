import os
import json
from collections import defaultdict

datadir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data')



TAX_IDS = {
    'human': '9606',
    'rat': '10116',
    'mouse': '10090',
    'rabbit': '9986',
    'hamster': '10030'
}


def load_homologene(species=None):
    if not species:
        species = set()

    gene_mapping = defaultdict(set)
    prev_cluster_id = None
    human_genes = set()
    other_genes = set()
    with open(os.path.join(datadir, "homologene.data")) as f:
        for line in f:
            line = line.strip()
            cluster_id, tax_id, gene_id = line.split('\t')[:3]

            if prev_cluster_id and cluster_id != prev_cluster_id:
                for other_gene in other_genes:
                    gene_mapping[other_gene].update(human_genes)
                human_genes = set()
                other_genes = set()

            if tax_id == '9606':
                human_genes.add(gene_id)
            if tax_id in species:
                other_genes.add(gene_id)

            prev_cluster_id = cluster_id

    for other_gene in other_genes:
        gene_mapping[other_gene].update(human_genes)

    return gene_mapping
