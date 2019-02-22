from collections import defaultdict
import re
from typing import List, Dict, Optional

import networkx as nx
import itertools

from tqdm import tqdm

import constants
import json
import logging

logging.basicConfig(level=logging.DEBUG)

PREFERED_IDS = ['uniprot', 'chebi']


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


def get_locations(entities: List[str], g: nx.MultiDiGraph) -> List[str]:
    locations = set()
    for entity in entities:
        for _, node, data in g.edges(data=True):
            if data['label'] == 'has_location':
                locations.add(node)

    return list(locations)


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


def add_event(events: List, text_expressions: Dict[str, dict], type: str) -> Dict:
    event_id = f'E{len(events)}'
    text_id = ensure_in_text_expressions(event_id, text_expressions, type)

    event = {'id': event_id, 'type': type, 'text_id': text_id}
    events.append(event)

    return event

def count_molucule_in_chemical(molecule: str, chemical: str):
    count = 0
    count += len(re.findall(f"(?:mono[-]?)?{molecule}", chemical))
    count += len(re.findall(f"di[-]?[s]?{molecule}", chemical)) * 2
    count += len(re.findall(f"tr[-]?i[s]?{molecule}", chemical)) * 3

    return count



def get_modification_type(left_mod: str, right_mod: str):
    if 'inactive' in left_mod and 'active' in right_mod and 'inactive' not in right_mod:
       return constants.ACTIVATION
    elif 'active' in left_mod and 'inactive' in right_mod and 'inactive' not in left_mod:
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
            event['text'] = f"{event['id']}\t{mod_type}:{event['text_id']} Theme:{theme_id}"
            logging.debug(f"Added event {event}")



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
            event['text'] = f"{event['id']}\t{constants.TRANSPORT}:{event['text_id']} ToLoc:{right_loc_text_id} FromLoc:{left_loc_text_id}"
            logging.debug(f"Added event {event}")


def reactions_to_standoff(reactions, g):
    text_expressions = defaultdict(dict)
    events = []

    id_to_entity = get_entities(reactions, g)
    for ent_id, ent_name in id_to_entity.items():
        text_expressions[ent_name]['id'] = f'T{len(text_expressions)}'
        text_expressions[ent_name]['type'] = get_entity_type(ent_id)

    for reaction in reactions:
        add_transports(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
        add_modifications(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)


if __name__ == '__main__':
    g = nx.MultiDiGraph()
    fname = 'data/reactome'
    graph = fname + '.cif'
    references = fname + '_references.json'

    with open(graph) as f:
        for line in f:
            e1, e2, r = line.strip().split('\t')
            g.add_edge(e1, e2, label=r)

    with open(references) as f:
        reactions_to_pm_ids = json.load(f)

    pm_id_to_reactions = defaultdict(list)
    for reaction, pm_ids in reactions_to_pm_ids.items():
        for pm_id in pm_ids:
            pm_id_to_reactions[pm_id].append(reaction)
    all_reactions = list(pm_id_to_reactions.values())
    for reactions in tqdm(all_reactions):
        reactions_to_standoff(reactions, g)
