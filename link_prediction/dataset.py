from collections import defaultdict

import numpy as np
import random
import torch
import math

from sklearn.utils import shuffle


class Dataset:
    def __init__(self, ds_name):
        self.name = ds_name
        self.dir = "data/" + ds_name + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.data = {spl: self.read(self.dir + spl + ".txt") for spl in ["train", "valid", "test"]}
        self.batch_index = defaultdict(int)
       
    def read(self, file_path):

        with open(file_path, "r") as f:
            lines = f.readlines()
        
        pairs_to_labels = defaultdict(set)

        for i, line in enumerate(lines):
            triple = line.strip().split("\t")
            if triple[1] == "NA":
                continue
            triple = self.triple2ids(triple)
            pairs_to_labels[(triple[0], triple[2])].add(triple[1])

        X = np.zeros((len(pairs_to_labels), 2))
        y = np.zeros((len(pairs_to_labels), self.num_rel))

        for i, (pair, labels) in enumerate(pairs_to_labels.items()):
            X[i, 0] = pair[0]
            X[i, 1] = pair[1]
            for label in labels:
                if label:
                    y[i, label] = 1

        X, y = shuffle(X, y)
        return [torch.tensor(X).long().cuda(), torch.tensor(y).long().cuda()]

    @property
    def num_ent(self):
        return len(self.ent2id)

    @property
    def num_rel(self):
        return len(self.rel2id)
                     
    def triple2ids(self, triple):
        return [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
                     
    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]
            
    def get_rel_id(self, rel):

        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]
                     
    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, self.num_ent - 1)
        return rand_ent
                     
    def next_batch(self, batch_size, split):
        if self.batch_index[split] + batch_size < len(self.data["train"][0]):
            X = self.data[split][0][self.batch_index[split]: self.batch_index[split] +batch_size]
            y = self.data[split][1][self.batch_index[split]: self.batch_index[split] + batch_size]
            self.batch_index[split] += batch_size
        else:
            X = self.data[split][0][self.batch_index[split]:]
            y = self.data[split][1][self.batch_index[split]:]

            perm = torch.randperm(self.data[split][0].size(0))
            self.data[split][0] = self.data[split][0][perm]
            self.data[split][1] = self.data[split][1][perm]

            self.batch_index[split] = 0
        return X, y
                     
    def was_last_batch(self, split):
        return (self.batch_index[split] == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))

