from collections import defaultdict
from typing import List, Dict, Optional

import networkx as nx
import itertools

from tqdm import tqdm

import constants
import json

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


def get_locations(g: nx.MultiDiGraph) -> List[str]:
    locations = set()
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


def add_transports(reaction: str, text_expressions: Dict[str, Dict],
                   entities: Dict[str, str], events: List[Dict], g: nx.MultiDiGraph):
    lefts = get_lefts(reaction, g)
    rights = get_rights(reaction, g)
    common_ents = set(entities[e] for e in lefts) & set(entities[e] for e in rights)

    left_locations = {}
    for left in lefts:
        name = entities[left]
        left_locations[name] = get_location(left)

    right_locations = {}
    for right in rights:
        name = entities[right]
        right_locations[name] = get_location(right)

    for ent in common_ents:
        left_loc = left_locations[ent]
        right_loc = right_locations[ent]
        if left_loc != right_loc:
            event_id = f'E{len(events)}'

            left_loc_text_id = text_expressions[left_loc]['id']
            right_loc_text_id = text_expressions[right_loc]['id']
            text_id = f'T{len(text_expressions)}'
            text_expressions[event_id]['id'] = text_id
            text_expressions[event_id]['type'] = f'{constants.TRANSPORT}'
            event = f'{event_id}\t{constants.TRANSPORT}:{text_id} ToLoc:{right_loc_text_id} FromLoc:{left_loc_text_id}'
            events.append({'id': event_id, 'text': event})


def reactions_to_standoff(reactions, g):
    text_expressions = defaultdict(dict)
    events = []

    id_to_entity = get_entities(reactions, g)
    for ent_id, ent_name in id_to_entity.items():
        text_expressions[ent_name]['id'] = f'T{len(text_expressions)}'
        text_expressions[ent_name]['type'] = get_entity_type(ent_id)
    locations = get_locations(g)

    for loc in locations:
        text_expressions[loc]['id'] = f'T{len(text_expressions)}'
        text_expressions[loc]['type'] = constants.LOCATION

    for reaction in reactions:
        if 'Transport' in reaction:
            add_transports(reaction, entities=id_to_entity, text_expressions=text_expressions, events=events, g=g)
            pass


if __name__ == '__main__':
    g = nx.MultiDiGraph()
    fname = 'data/ncbi'
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
