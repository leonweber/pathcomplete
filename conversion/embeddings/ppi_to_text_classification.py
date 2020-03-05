import argparse
from collections import defaultdict, namedtuple

from lxml import etree
from pathlib import Path


Entity = namedtuple('Entity', 'name position') # position is position in sentence text (start, end)

def get_entity_position( entity):
    char_offsets = entity.xpath('./@charOffset')[0].split(',')
    char_offset = tuple(int(i) for i in char_offsets[0].split('-'))
    return char_offset

def get_all_entity_pairs(sentence):
    """
    Extract all entity pairs from sentence

    If multiple positions are given for an entity, the first one is selected.

    :param sentence:
    :return: list of entity pairs, list of labels (True if interaction between entities, False otherwise)
    """
    entities = {}
    labels = []
    entity_pairs = []

    for entity in sentence.xpath('.//entity'):
        entity_position = get_entity_position(entity)
        text = entity.xpath('./@text')[0]
        entity_id = entity.xpath('./@id')[0]
        entities[entity_id] = Entity(text, entity_position)

    for interaction in sentence.xpath('.//interaction'):
        entity1 = entities[interaction.xpath('.//@e1')[0]]
        entity2 = entities[interaction.xpath('.//@e2')[0]]

        entity_pairs.append((entity1, entity2))
        labels.append(True)

    for non_interaction in sentence.xpath('.//non_interaction'):
        entity1 = entities[non_interaction.xpath('.//@e1')[0]]
        entity2 = entities[non_interaction.xpath('.//@e2')[0]]

        entity_pairs.append((entity1, entity2))
        labels.append(False)

    return entity_pairs, labels

def xml_to_df(xml):
    tree = etree.parse(xml)
    df = defaultdict(list)
    docs = tree.xpath('//document')
    for doc in docs:
        doc_id = doc.attrib['id']
        for sent in doc.xpath('sentences'):
            text = sent.text()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

