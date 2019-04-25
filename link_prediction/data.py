from pathlib import Path

import numpy as np
from dgl.contrib.data.knowledge_graph import _read_dictionary, _read_triplets_as_list


class RGCNLinkDataset:
    """
    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    train: numpy.array
        all relation triplets (src, rel, dst) for training
    valid: numpy.array
        all relation triplets (src, rel, dst) for validation
    test: numpy.array
        all relation triplets (src, rel, dst) for testing

    """
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        train_path = self.path/'train.txt'
        valid_path = self.path/'valid.txt'
        test_path = self.path/'test.txt'
        entity_path = self.path/'entities.dict'
        relation_path = self.path/'relations.dict'
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        self.valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# edges: {}".format(len(self.train)))
