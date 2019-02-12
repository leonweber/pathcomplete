import itertools
import json
import sys

import networkx as nx
import pandas as pd
from collections import defaultdict


DEGRADATION = 'Degradation'
DEPHOSPHORYLATION = 'Dephosphorylation'
PHOSPHORYLATION = 'Phosphorylation'
BINDING = 'Binding'
DISSOCIATION = 'Dissociation'

with open('data/uniprot_to_hgnc.json') as f:
    uniprot_to_hgnc = json.load(f)


class Tabularizer:

    def __init__(self, g, id_to_references):
        self.g = g
        self.id_to_references = id_to_references

    def denormalize_protein(self, protein):
        for _, id_, data in self.g.edges(protein, data=True):
            if data['label'] == 'has_id':
                uniprot = id_.split('/')[-1]

                if uniprot in uniprot_to_hgnc:
                    return uniprot_to_hgnc[uniprot][0]
                else:
                    return protein
        return protein


    def get_location(self, n):
        for _, v, data in self.g.edges(n, data=True):
            if data['label'] == 'has_location':
                return v
        else:
            return None


    def get_modification(self, n):
        for _, v, data in self.g.edges(n, data=True):
            if data['label'] == 'has_modification':
                return v
        else:
            return None

    def get_reaction_type(self, r):
        left_phosphos = ['phospho' in l for l in r['lefts']]
        right_phosphos = ['phospho' in l for l in r['rights']]
        if len(r['lefts']) > 0 and len(r['rights']) == 0:
            return DEGRADATION
        elif sum(left_phosphos) > sum(right_phosphos):
            return DEPHOSPHORYLATION
        elif sum(left_phosphos) < sum(right_phosphos):
            return PHOSPHORYLATION
        elif 'complex' in r['id'].lower() or len(r['lefts']) > len(r['rights']):
            return BINDING
        elif len(r['lefts']) < len(r['rights']):
            return DISSOCIATION
        else:
            return r['id']


    def get_reaction(self, reaction, query_proteins=None):
        query_proteins = query_proteins or set()
        lefts = []
        rights = []
        all_participants = []
        inhibitors = []
        regulators = []
        activators = []
        references = []
        if reaction in self.id_to_references:
            references = self.id_to_references[reaction]

        for r, n, data in self.g.edges(reaction, data=True):
            for desc in nx.descendants(self.g, n):
                if 'uniprot' in desc:
                    all_participants.append(desc.split('/')[-1])

            if data['label'] == 'has_left':
                lefts.append(self.flatten_complex(n))
            elif data['label'] == 'has_right':
                rights.append(self.flatten_complex(n))
            elif data['label'] == 'inhibition':
                inhibitors.append(self.flatten_complex(n))
            elif data['label'] == 'regulation':
                regulators.append(self.flatten_complex(n))
            elif data['label'] == 'activation':
                activators.append(self.flatten_complex(n))
            else:
                print(data['label'])
        n_hits = len(set(all_participants) & set(query_proteins))
        lefts = list(filter(None, lefts))
        rights = list(filter(None, rights))
        inhibitors = list(filter(None, inhibitors))
        regulators = list(filter(None, regulators))
        activators = list(filter(None, activators))

        return {'lefts': lefts, 'rights': rights, 'participants': all_participants, 'id': reaction,
                'inhibitors': inhibitors, 'regulators': regulators, 'activators': activators, 'references': references,
                'n_hits': n_hits}

    def flatten_complex(self, c):
        #     location = get_location(c)
        modification = self.get_modification(c)
        if 'Protein' in c:
            out = self.denormalize_protein(c)
        else:
            members = list(self.g.neighbors(c))
            members = [self.flatten_complex(m) for m in members if m]
            members = sorted([m for m in members if m])
            out = ':'.join(members)
        #     if location:
        #         out = out + f"({location})"
        if modification:
            out = out + f"({modification})"
        return out

    def reactions_to_df(self, reactions):
        df = defaultdict(list)
        for r in reactions:
            df['type'].append(self.get_reaction_type(r))
            df['lhs'].append('|'.join(r['lefts']))
            df['rhs'].append('|'.join(r['rights']))
            df['activators'].append('|'.join(r['activators']))
            df['inhibitors'].append('|'.join(r['inhibitors']))
            df['regulators'].append('|'.join(r['regulators']))
            df['references'].append('|'.join(r['references']))
            df['n_hits'].append(r['n_hits'])
        df = pd.DataFrame(df).sort_values('n_hits', ascending=False)

        return df

    def get_reactions(self, query_proteins=None):
        reactions = [self.get_reaction(n, query_proteins) for n in self.g if "R-HSA" in n or "Transport" in n or "ComplexAssembly" in n or "Reaction" in n]
        if query_proteins:
            reactions = [r for r in reactions if r['n_hits'] > 0]
        return reactions


if __name__ == '__main__':
    g = nx.MultiDiGraph()
    #fname = sys.argv[1]
    fname = 'data/reactome'
    graph = fname + '.cif'
    references = fname + '_references.json'
    query_graph = pd.read_csv('data/nfkappab.tsv', sep='\t')

    proteins = set()
    for complx in itertools.chain(query_graph['Protein A'], query_graph['Protein B']):
        proteins.update(complx.split(':'))

    with open(graph) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            g.add_edge(e1, e2, label=r)

    with open(references) as f:
        references = json.load(f)

    tab = Tabularizer(g, references)
    reactions = tab.get_reactions(query_proteins=proteins)
    df = tab.reactions_to_df(reactions).to_csv('test.csv', index=False)




