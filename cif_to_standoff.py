import argparse
import os
import sys
from collections import defaultdict
import re
from typing import List, Dict, Optional

import networkx as nx
import itertools

from pathlib import Path
from tqdm import tqdm

import constants
import json
import logging

logging.basicConfig(level=logging.INFO)

PREFERED_IDS = ['uniprot', 'chebi']


class Event:
    def __init__(self, id_, type_, text_id):
        self.themes = []
        self.products = []
        self.cause = None
        self.from_loc = None
        self.to_loc = None
        self.id = id_
        self.type = type_
        self.text_id = text_id

    def to_text(self):
        text = f"{self.id}\t{self.type}:{self.text_id}"
        for i, theme in enumerate(self.themes, start=1):
            text += f" Theme{i}:{theme}" if i > 0 else f" Theme:{theme}"
        for i, product in enumerate(self.products, start=1):
            text += f" Product{i}:{product}" if i > 0 else f" Product:{product}"

        if self.cause:
            text += f" Cause: {self.cause}"

        if self.from_loc:
            text += f" FromLoc: {self.from_loc}"

        if self.to_loc:
            text += f" ToLoc: {self.to_loc}"

        return text

    def __str__(self):
        return self.to_text()




def get_entity_type(node: str) -> Optional[str]:
    if '#Protein' in node:
        return constants.GENE_OR_GENE_PRODUCT
    elif '#PhysicalEntity' in node:
        return constants.ENTITY
    elif '#Complex' in node:
        return constants.COMPLEX
    elif '#SmallMolecule' in node:
        return constants.CHEMICAL
    elif '#Dna' in node:
        return constants.GENE_OR_GENE_PRODUCT
    elif '#Rna' in node:
        return constants.GENE_OR_GENE_PRODUCT
    else:
        return None


def get_complex_members(complex: str, g: nx.MultiDiGraph) -> List[str]:
    members = []
    for _, node, data in g.edges(complex, data=True):
        if data['label'] == 'has_component':
            members.append(node)
    assert len(members) > 1
    return members


def get_entity_name(entity: str, g: nx.MultiDiGraph) -> str:
    names = []
    for _, node, data in g.edges(entity, data=True):
        if data['label'] == 'has_id':
            names.append(node)
    for name in names:
        for prefered_id in PREFERED_IDS:
            if prefered_id in name:
                return name
    else:
        if len(names) > 0:
            return names[0]
        else:
            if 'Complex' in entity:
                members = []
                for member in get_complex_members(entity, g):
                    member_name = get_entity_name(member, g)
                    members.append(member_name)
                return ':'.join(members)


def get_entities(reactions: List[str], g: nx.MultiDiGraph) -> Dict[str, str]:
    entities = {}
    for reaction in reactions:
        for _, node, data in g.edges(reaction, data=True):
            if not get_entity_type(node):
                continue
            name = get_entity_name(node, g)
            if node not in entities:
                entities[node] = name

    return entities


def get_lefts(reaction: str, g: nx.MultiDiGraph) -> List[str]:
    lefts = []
    for _, node, data in g.edges(reaction, data=True):
        if data['label'] == 'has_left':
            lefts.append(node)
    return lefts


def get_rights(reaction: str, g: nx.MultiDiGraph) -> List[str]:
    rights = []
    for _, node, data in g.edges(reaction, data=True):
        if data['label'] == 'has_right':
            rights.append(node)
    return rights


def get_location(entity: str, g: nx.MultiDiGraph):
    for _, node, data in g.edges(entity, data=True):
        if data['label'] == 'has_location':
            return node
    else:
        return None

def get_modification(entity:str, g: nx.MultiDiGraph):
    for _, node, data in g.edges(entity, data=True):
        if data['label'] == 'has_modification':
            return node
    else:
        return None

def ensure_in_text_expressions(key: str, text_expressions: Dict[str, dict], type: str) -> str:
    if key not in text_expressions:
        text_expressions[key]['id'] = f'T{len(text_expressions)}'
        text_expressions[key]['type'] = type
    return text_expressions[key]['id']


def add_event(events: List, text_expressions: Dict[str, dict], type: str) -> Event:
    """
    Add new event to events. Returns the dict comprising the new event

    :param events:
    :param text_expressions:
    :param type:
    :return:
    """
    event_id = f'E{len(events)}'
    text_id = ensure_in_text_expressions(event_id, text_expressions, type)

    event = Event(id_=event_id, type_= type, text_id=text_id)
    events.append(event)

    return event


def count_molucule_in_chemical(molecule: str, chemical: str):
    count = 0
    count += len(re.findall(f"(?:mono[-]?)?{molecule}", chemical))
    count += len(re.findall(f"di[-]?[s]?{molecule}", chemical)) * 2
    count += len(re.findall(f"tr[-]?i[s]?{molecule}", chemical)) * 3

    return count


def get_modification_type(left_mod: str, right_mod: str):
    if ('inactive' in left_mod or not 'active' in left_mod) and 'active' in right_mod and 'inactive' not in right_mod:
       return constants.ACTIVATION
    elif 'active' in left_mod and ('inactive' in right_mod or not 'active' in right_mod) and 'inactive' not in left_mod:
        return constants.INACTIVATION
    elif 'sumoylated' not in left_mod and 'sumoylated' in right_mod:
        return constants.SUMOYLATION
    elif 'sumoylated' in left_mod and 'sumoylated' not in right_mod:
        return constants.DESUMOYLATION
    elif count_molucule_in_chemical('phosph', left_mod) < count_molucule_in_chemical('phosph', right_mod):
        return constants.PHOSPHORYLATION
    elif count_molucule_in_chemical('phosph', left_mod) > count_molucule_in_chemical('phosph', right_mod):
        return constants.DEPHOSPHORYLATION
    elif 'ubiquitin' not in left_mod and 'ubiquitin' in right_mod:
        return constants.UBIQUITINYLATION
    elif 'ubiquitin' in left_mod and 'ubiquitin' not in right_mod:
        return constants.DEUBIQUITINYLATION
    elif count_molucule_in_chemical('acetyl', left_mod) < count_molucule_in_chemical('acetyl', right_mod):
        return constants.ACETYLATION
    elif count_molucule_in_chemical('acetyl', left_mod) > count_molucule_in_chemical('acetyl', right_mod):
        return constants.DEACETYLATION
    elif count_molucule_in_chemical('methyl', left_mod) < count_molucule_in_chemical('methyl', right_mod):
        return constants.METHYLATION
    elif count_molucule_in_chemical('methyl', left_mod) > count_molucule_in_chemical('methyl', right_mod):
        return constants.DEMETHYLATION
    else:
        logging.warning(f"Could not determine modification: {left_mod} -> {right_mod}")
        return None


def add_regulations(reaction: str, event_id: str, text_expressions: Dict[str, Dict],
                    entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):

    for _, node, data in g.edges(reaction, data=True):
        if data['label'] in {'activation', 'inhibition', 'regulation'}:
            regulator = entities[node]
            if data['label'] == 'activation':
                event_type = constants.POSITIVE_REGULATION
            elif data['label'] == 'inhibition':
                event_type = constants.NEGATIVE_REGULATION
            elif data['label'] == 'regulation':
                event_type = constants.REGULATION
            else:
                raise ValueError

            theme_id = text_expressions[event_id]['id']
            cause_id = text_expressions[regulator]['id']

            event = add_event(events, text_expressions, event_type)
            event.cause = cause_id
            event.themes.append(theme_id)


def add_modifications(reaction: str, text_expressions: Dict[str, Dict],
                      entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):
    lefts = get_lefts(reaction, g)
    rights = get_rights(reaction, g)
    common_ents = set(entities[e] for e in lefts) & set(entities[e] for e in rights)
    left_modifications = {}
    for left in lefts:
        name = entities[left]
        left_modifications[name] = get_modification(left, g)

    right_modifications = {}
    for right in rights:
        name = entities[right]
        right_modifications[name] = get_modification(right, g)

    for ent in common_ents:
        left_mod = left_modifications[ent] or ''
        right_mod = right_modifications[ent] or ''
        if left_mod != right_mod:
            mod_type = get_modification_type(left_mod, right_mod)
            if not mod_type:
                continue

            theme_id = text_expressions[ent]['id']
            event = add_event(events, text_expressions, mod_type)
            event.themes.append(theme_id)
            logging.debug(f"Added event {event}")
            add_regulations(reaction, event.id, text_expressions=text_expressions, entities=entities, events=events, g=g)



def add_transports(reaction: str, text_expressions: Dict[str, Dict],
                   entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):
    lefts = get_lefts(reaction, g)
    rights = get_rights(reaction, g)
    common_ents = set(entities[e] for e in lefts) & set(entities[e] for e in rights)

    left_locations = {}
    for left in lefts:
        name = entities[left]
        left_locations[name] = get_location(left, g)

    right_locations = {}
    for right in rights:
        name = entities[right]
        right_locations[name] = get_location(right, g)

    for ent in common_ents:
        left_loc = left_locations[ent]
        right_loc = right_locations[ent]
        if left_loc != right_loc:
            left_loc_text_id = ensure_in_text_expressions(left_loc, text_expressions, constants.LOCATION)
            right_loc_text_id = ensure_in_text_expressions(right_loc, text_expressions, constants.LOCATION)

            event = add_event(events, text_expressions, constants.TRANSPORT)
            event.from_loc = left_loc_text_id
            event.to_loc = right_loc_text_id
            logging.debug(f"Added event {event}")
            add_regulations(reaction, event.id, text_expressions=text_expressions, entities=entities, events=events, g=g)


def add_degradations(reaction: str, text_expressions: Dict[str, Dict],
                     entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):
    lefts = get_lefts(reaction, g)
    rights = get_rights(reaction, g)

    if len(rights) == 0:
        for left in lefts:
            name = entities[left]
            event = add_event(events, text_expressions, constants.DEGRADATION)
            event.themes.append(text_expressions[name])

            logging.debug(f"Added event {event}")
            add_regulations(reaction, event.id, text_expressions=text_expressions, entities=entities, events=events, g=g)


def add_bindings(reaction: str, text_expressions: Dict[str, Dict],
                 entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):
    lefts = get_lefts(reaction, g)
    rights = get_rights(reaction, g)
    if len(lefts) > len(rights) and len(rights) == 1 and 'Complex' in rights[0]:
        product_name = entities[rights[0]]
        event = add_event(events, text_expressions, constants.BINDING)
        event.products.append(text_expressions[product_name]['id'])
        event.themes = [text_expressions[entities[left]]['id'] for left in lefts]

        logging.debug(f"Added event {event}")
        add_regulations(reaction, event.id, text_expressions=text_expressions, entities=entities, events=events, g=g)


def add_dissociations(reaction: str, text_expressions: Dict[str, Dict],
                     entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):
    lefts = get_lefts(reaction, g)
    rights = get_rights(reaction, g)
    if len(lefts) < len(rights) and len(lefts) == 1 and 'Complex' in lefts[0]:
        event = add_event(events, text_expressions, constants.DISSOCIATION)
        theme_id = text_expressions[entities[lefts[0]]]['id']
        event.themes.append(theme_id)
        event.products = [text_expressions[entities[right]]['id'] for right in rights]

        logging.debug(f"Added event {event}")
        add_regulations(reaction, event.id, text_expressions, entities, events, g)

def reactions_to_standoff(reactions, g):
    text_expressions = defaultdict(dict)
    events = []

    id_to_entity = get_entities(reactions, g)
    for ent_id, ent_name in id_to_entity.items():
        text_expressions[ent_name]['id'] = f'T{len(text_expressions)}'
        text_expressions[ent_name]['type'] = get_entity_type(ent_id)

    for reaction in reactions:
        n_events_before = len(events)
        add_transports(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        add_modifications(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        add_degradations(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        add_bindings(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        add_dissociations(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        # add_gene_expressions(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        # add_transcriptions(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        # add_translations(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        n_events_after = len(events)

        if n_events_before == n_events_after:
            logging.warning(f"Did not add any events during processing of {reaction}")

    return events, text_expressions


def mention_to_str(name, info):
    return f"{info['id']}\t{info['type']} ? ?\t{name}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    g = nx.MultiDiGraph()
    fname = args.input
    graph = fname + '.cif'
    references = fname + '_references.json'

    with open(graph) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            g.add_edge(e1, e2, label=r)

    with open(references) as f:
        reactions_to_pm_ids = json.load(f)

    pm_id_to_reactions = defaultdict(list)
    for reaction, pm_ids in list(reactions_to_pm_ids.items()):
        for pm_id in pm_ids:
            pm_id_to_reactions[pm_id].append(reaction)

    pm_id_to_events = {}
    for pm_id, reactions in tqdm(pm_id_to_reactions.items()):
        events, text_mentions = reactions_to_standoff(reactions, g)
        if len(events) > 0:
            pm_id_to_events[pm_id] = (events, text_mentions)

    n_events = 0
    for events in pm_id_to_events.values():
        n_events += len(events)
    logging.info(f"Collected {n_events} events spread across {len(pm_id_to_events)} articles.")

    out_dir = Path(args.output)
    os.makedirs(out_dir, exist_ok=True)
    for pm_id, (events, text_mentions) in pm_id_to_events.items():
        pm_id = pm_id.split('/')[-1]
        with open(str(out_dir/pm_id) + '.a1', 'w') as f_a1, open(str(out_dir/pm_id) + '.a2', 'w') as f_a2:
            for name, info in text_mentions.items():
                if name and name.startswith('E'):
                    f_a2.write(mention_to_str(name, info))
                    f_a2.write("\n")
                else:
                    f_a1.write(mention_to_str(name, info))
                    f_a1.write("\n")

            for event in events:
                f_a2.write(event.to_text())
                f_a2.write('\n')

