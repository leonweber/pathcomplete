import itertools
import json
import re
import sys

import networkx as nx
import pandas as pd
from collections import defaultdict

DEGRADATION = 'Degradation'
DEPHOSPHORYLATION = 'Dephosphorylation'
PHOSPHORYLATION = 'Phosphorylation'
UBIQUITINYLATION = 'Ubiquitinylation'
DEUBIQUITINYLATION = 'Deubiquitinylation'
METHYLATION = 'Methylation'
DEMETHYLATION = 'Demethylation'
ACETYLATION = 'Acetylation'
DEACETYLATION = 'Deacetylation'
BINDING = 'Binding'
DISSOCIATION = 'Dissociation'
TRANSCRIPTION = 'Transcription'
GENE_EXPRESSION = 'Gene_expression'
CONVERSION = 'Conversion'
ACTIVATION = 'Activation'
DEACTIVATION = 'Deactivation'
TRANSPORT = 'Transport'
POSITIVE_REGULATION = 'Positive_regulation'
NEGATIVE_REGULATION = 'Negative_regulation'
REGULATION = 'Regulation'

GOGP = 'Gene_or_gene_product'
COMPLEX = 'Complex'

with open('data/uniprot_to_hgnc.json') as f:
    uniprot_to_hgnc = json.load(f)


def requires_right(type_):
    return type_ in {CONVERSION}


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
                    return id_
        return protein

    def denormalize_dna(self, dna):
        for _, id_, data in self.g.edges(dna, data=True):
            if data['label'] == 'has_id':
                return 'dna_' + id_
        return dna

    def denormalize_rna(self, rna):
        for _, id_, data in self.g.edges(rna, data=True):
            if data['label'] == 'has_id':
                return 'rna_' + id_
        return rna

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

    def get_reaction_types(self, r):
        types = []

        left_phosphos = [l.count('phospho') for l in r['lefts']]
        right_phosphos = [l.count('phospho') for l in r['rights']]

        left_ubiquis = [l.count('ubiqui') for l in r['lefts']]
        right_ubiquis = [l.count('ubiqui') for l in r['rights']]

        left_methyls = [l.count('methyl') for l in r['lefts']]
        right_methyls = [l.count('methyl') for l in r['rights']]

        left_acetyl = [l.count('acetyl') for l in r['lefts']]
        right_acetyl = [l.count('acetyl') for l in r['rights']]

        left_active = [l.count('active') for l in r['lefts']]
        right_active = [l.count('active') for l in r['rights']]

        left_locs = []
        right_locs = []

        for l in r['lefts']:
            left_locs += re.findall(r'\(LOC:(.*?)\)', l)

        for l in r['rights']:
            right_locs += re.findall(r'\(LOC:(.*?)\)', l)

        if 'ComplexAssembly' in r['id']:
            types.append(BINDING)
        if 'Transport' in r['id'] and 'Reaction' not in r['id']:
            types.append(TRANSPORT)

        if len(r['lefts']) > 0 and len(r['rights']) == 0:
            types.append(DEGRADATION)

        if any(['dna' in l for l in r['lefts']]):
            if any(['rna' in l for l in r['rights']]):
                types.append(TRANSCRIPTION)
            else:
                types.append(GENE_EXPRESSION)

        if sum(left_phosphos) > sum(right_phosphos):
            types.append(DEPHOSPHORYLATION)
        if sum(left_phosphos) < sum(right_phosphos):
            types.append(PHOSPHORYLATION)

        if sum(left_ubiquis) > sum(right_ubiquis):
            types.append(DEUBIQUITINYLATION)
        if sum(left_ubiquis) < sum(right_ubiquis):
            types.append(UBIQUITINYLATION)

        if sum(left_methyls) > sum(right_methyls):
            types.append(DEMETHYLATION)
        if sum(left_methyls) < sum(right_methyls):
            types.append(METHYLATION)

        if sum(left_acetyl) > sum(right_acetyl):
            types.append(DEACETYLATION)
        if sum(left_acetyl) < sum(right_acetyl):
            types.append(ACETYLATION)

        if sum(left_active) > sum(right_active):
            types.append(DEACTIVATION)
        if sum(left_active) < sum(right_active):
            types.append(ACTIVATION)

        if len(r['lefts']) > len(r['rights']):
            types.append(BINDING)
        if len(r['lefts']) < len(r['rights']):
            types.append(DISSOCIATION)
        if left_locs != right_locs:
            types.append(TRANSPORT)
        if len(types) == 0:
            types.append(CONVERSION)

        return types

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

        reaction = {'lefts': lefts, 'rights': rights, 'participants': all_participants, 'id': reaction,
                    'inhibitors': inhibitors, 'regulators': regulators, 'activators': activators,
                    'references': references,
                    'n_hits': n_hits}
        reaction['types'] = self.get_reaction_types(reaction)
        return reaction

    def flatten_complex(self, c):
        location = self.get_location(c)
        modification = self.get_modification(c)
        if 'Protein' in c:
            out = self.denormalize_protein(c)
        elif 'Dna' in c:
            out = self.denormalize_dna(c)
        elif 'Rna' in c:
            out = self.denormalize_rna(c)
        else:
            members = list(self.g.neighbors(c))
            members = [self.flatten_complex(m) for m in members if m]
            members = sorted([m for m in members if m])
            out = ':'.join(members)
            if location and out:
                out = out + f"(LOC:{location})"
        if modification:
            out = out + f"({modification})"
        return out

    def reactions_to_df(self, reactions):
        df = defaultdict(list)
        for r in reactions:
            df['types'].append('#'.join(self.get_reaction_types(r)))
            df['lhs'].append('#'.join(r['lefts']))
            df['rhs'].append('#'.join(r['rights']))
            df['activators'].append('#'.join(r['activators']))
            df['inhibitors'].append('#'.join(r['inhibitors']))
            df['regulators'].append('#'.join(r['regulators']))
            df['references'].append('#'.join(r['references']))
            df['id'].append(r['id'])
            df['n_hits'].append(r['n_hits'])
        df = pd.DataFrame(df).sort_values('n_hits', ascending=False)

        return df

    def resolve_complex(self, complex, entities, events):
        members = []
        complex = re.sub(r'\(.*?\)', '', complex)
        complex = complex.replace('dna_', '')
        complex = complex.replace('rna_', '')
        for member in complex.split(':'):
            members.append(member)
        members = sorted(members)

        complex = ':'.join(members)
        if complex in entities:
            return entities[complex]
        else:
            themes = []
            for member in members:
                if member not in entities:
                    entity = {'id': 'T' + str(len(entities)), 'type': GOGP, 'name': member}
                    entities[member] = entity
                themes.append(entities[member])

            if len(members) == 1:
                return entities[members[0]]

            entities[complex] = {'id': 'T' + str(len(entities)), 'type': COMPLEX, 'name': complex}
            event = {'themes': themes, 'type': BINDING, 'id': 'E' + str(len(events))}
            events[event['id']] = event
        return entities[complex]

    def reactions_to_standoff(self, reactions):
        events_by_pmid = defaultdict(list)
        for r in reactions:
            entities = {}
            events = {}
            for type_ in r['types']:
                themes = []
                event = {'type': type_}

                for left in r['lefts']:
                    left_ent = self.resolve_complex(left, entities, events)
                    themes.append(left_ent)
                if requires_right(type_):
                    products = []
                    for right in r['rights']:
                        right_ent = self.resolve_complex(right, entities, events)
                        products.append(right_ent)
                else:
                    products = None
                event['themes'] = themes
                if products:
                    event['products'] = products

                event['id'] = 'E' + str(len(events))
                events[event['id']] = event

                if len(r['activators']) == 1:
                    event['cause'] = self.resolve_complex(r['activators'][0], entities, events)
                elif len(r['activators']) > 1:
                    for activator in r['activators']:
                        positive_regulation_event = {'theme': event['id'], 'type': POSITIVE_REGULATION,
                                                     'cause': self.resolve_complex(activator, entities, events),
                                                     'id': 'E' + str(len(events))}
                        events[positive_regulation_event['id']] = positive_regulation_event
                    for inhibitor in r['inhibitors']:
                        negative_regulation_event = {'theme': event['id'], 'type': NEGATIVE_REGULATION,
                                                     'cause': self.resolve_complex(inhibitor, entities, events),
                                                     'id': 'E' + str(len(events))}
                        events[negative_regulation_event['id']] = negative_regulation_event
                    for regulator in r['regulators']:
                        regulation_event = {'theme': event['id'], 'type': REGULATION,
                                            'cause': self.resolve_complex(regulator, entities, events),
                                            'id': 'E' + str(len(events))}
                        events[regulation_event['id']] = regulation_event

            if len(entities) > 0 and len(events) > 0:
                for pm_id in self.id_to_references.get(r['id'], []):
                    events_by_pmid[pm_id].append({'entities': entities, 'events': events})

        return events_by_pmid




    def get_reactions(self, query_proteins=None):
        reactions = [self.get_reaction(n, query_proteins) for n in self.g if
                     "R-HSA" in n or "Transport" in n or "ComplexAssembly" in n or "Reaction" in n]
        if query_proteins:
            reactions = [r for r in reactions if r['n_hits'] > 0]
        return reactions


if __name__ == '__main__':
    g = nx.MultiDiGraph()
    # fname = sys.argv[1]
    fname = 'data/reactome'
    graph = fname + '.cif'
    references = fname + '_references.json'
    # query_graph = pd.read_csv('data/nfkappab.tsv', sep='\t')
    #
    # proteins = set()
    # for complx in itertools.chain(query_graph['Protein A'], query_graph['Protein B']):
    #     proteins.update(complx.split(':'))
    #
    with open(graph) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            g.add_edge(e1, e2, label=r)

    with open(references) as f:
        references = json.load(f)

    tab = Tabularizer(g, references)
    # reactions = tab.get_reactions(query_proteins=proteins)
    reactions = tab.get_reactions(query_proteins=None)
    standoff = tab.reactions_to_standoff(reactions[:100])
    # df = tab.reactions_to_df(reactions).to_csv(sys.argv[2], index=False, sep='\t')
