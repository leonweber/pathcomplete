import os

import numpy as np
import random
import torch
import math

class Dataset:
    def __init__(self, ds_name):
        self.name = ds_name
        self.dir = "data/" + ds_name + "/"

        self.ent2id = {}
        with open(os.path.join(self.dir, 'entities.dict')) as f:
            for line in f:
                id_, ent = line.strip().split('\t')
                self.ent2id[ent] = int(id_)

        self.rel2id = {}
        with open(os.path.join(self.dir, 'relations.dict')) as f:
            for line in f:
                id_, rel = line.strip().split('\t')
                self.rel2id[rel] = int(id_)
            assert self.rel2id["NA"] == 0

        self.data = {spl: self.read(self.dir + spl + ".txt") for spl in ["train", "valid", "test"]}
        self.batch_index = 0
       
    def read(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        triples = np.zeros((len(lines), 3))
        for i, line in enumerate(lines):
            triple = line.strip().split("\t")
            if triple[1] == "NA":
                continue
            triples[i] = np.array(self.triple2ids(triple))
        return triples

    @property
    def num_ent(self):
        return len(self.ent2id)

    @property
    def num_rel(self):
        return len(self.rel2id)
                     
    def triple2ids(self, triple):
        return [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
                     
    def get_ent_id(self, ent):
        return self.ent2id[ent]
            
    def get_rel_id(self, rel):
        return self.rel2id[rel]
                     
    def rand_ent_except(self, ent):
        rand_ent = random.randint(0, self.num_ent - 1)
        while(rand_ent == ent):
            rand_ent = random.randint(0, self.num_ent - 1)
        return rand_ent
                     
    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.data["train"]):
            batch = self.data["train"][self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.data["train"][self.batch_index:]
            self.batch_index = 0
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype("int") #appending the +1 label
                     
    def generate_neg(self, pos_batch, neg_ratio):
        neg_batch = np.repeat(np.copy(pos_batch), neg_ratio, axis=0)
        for i in range(len(neg_batch)):
            if random.random() < 0.5:
                neg_batch[i][0] = self.rand_ent_except(neg_batch[i][0]) #flipping head
            else:
                neg_batch[i][2] = self.rand_ent_except(neg_batch[i][2]) #flipping tail
        neg_batch[:,-1] = -1
        return neg_batch

    def next_batch(self, batch_size, neg_ratio, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch = self.generate_neg(pos_batch, neg_ratio)
        batch = np.append(pos_batch, neg_batch, axis=0)
        np.random.shuffle(batch)
        heads  = torch.tensor(batch[:,0]).long().to(device)
        rels   = torch.tensor(batch[:,1]).long().to(device)
        tails  = torch.tensor(batch[:,2]).long().to(device)
        labels = torch.tensor(batch[:,3]).float().to(device)
        return heads, rels, tails, labels
    
    def was_last_batch(self):
        return (self.batch_index == 0)

    def num_batch(self, batch_size):
        return int(math.ceil(float(len(self.data["train"])) / batch_size))

