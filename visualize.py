import json
import networkx as nx
import pandas as pd
from collections import defaultdict

with open('./uniprot_to_hgnc.json') as f:
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


    def get_participants(self, reaction):
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
        return {'lefts': lefts, 'rights': rights, 'participants': all_participants, 'id': reaction,
                'inhibitors': inhibitors, 'regulators': regulators, 'activators': activators, 'references': references}

    def flatten_complex(self, c):
        #     location = get_location(c)
        modification = self.get_modification(c)
        if 'Protein' in c:
            out = self.denormalize_protein(c)
        else:
            members = list(self.g.neighbors(c))
            members = [self.flatten_complex(m) for m in members if m]
            members = [m for m in members if m]
            out = ':'.join(members)
        #     if location:
        #         out = out + f"({location})"
        if modification:
            out = out + f"({modification})"
        return out

    def reactions_to_df(self, reactions):
        df = defaultdict(list)
        for r in reactions:
            df['type'].append(r['id'].split('/')[-1])
            df['lhs'].append(','.join(r['lefts']))
            df['rhs'].append(','.join(r['rights']))
            df['activators'].append(','.join(r['activators']))
            df['inhibitors'].append(','.join(r['inhibitors']))
            df['regulators'].append(','.join(r['regulators']))
            df['references'].append(','.join(r['references']))
        return pd.DataFrame(df)

    def get_reactions(self):
        reactions = [self.get_participants(n) for n in self.g if "R-HSA" in n or "Transport" in n or "ComplexAssembly" in n or "Reaction" in n]
        return reactions


if __name__ == '__main__':
    g = nx.MultiDiGraph()
    with open('data/reactome.cif') as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            g.add_edge(e1, e2, label=r)

    tab = Tabularizer(g)
    reactions = tab.get_reactions()




