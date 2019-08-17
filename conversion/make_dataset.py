import argparse
import json
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import mygene
import itertools
import numpy as np
import re

import evex_to_sifnx_preds

INTERACTION_TYPES = set(evex_to_sifnx_preds.TYPE_MAPPING.values())

def augment_interactions(interactions):
    augmented_interactions = {}
    for i in interactions:
        augmented_interactions[i] = interactions[i]
        e1, r, e2 = i.split(',')
        if r in {'controls-phosphorylation-of', 'controls-transport-of'}:
            augmented_interactions[f"{e1},controls-state-change-of,{e2}"] = interactions[i]
        if r == 'in-complex-with':
            augmented_interactions[f"{e2},in-complex-with,{e1}"] = interactions[i]

    return augmented_interactions


def subsample_genes(df, size):
    neighbours = defaultdict(list)
    for _, row in df.iterrows():
        prot_a = row['PARTICIPANT_A']
        prot_b = row['PARTICIPANT_B']

        if ':' in prot_a or ':' in prot_b:
            continue

        neighbours[prot_a].append(prot_b)
        neighbours[prot_b].append(prot_a)
    
    np.random.seed(5005)
    selected_nodes = set(np.random.choice(sorted(neighbours), 5, replace=False))

    while len(selected_nodes) < size:
        active_node = np.random.choice(list(selected_nodes), 1)[0]
        if neighbours[active_node]:
            random_neighbour = np.random.choice(neighbours[active_node], 1)[0]
            selected_nodes.add(random_neighbour)
    
    return selected_nodes



def hgnc_to_uniprot(symbol, mapping, mg):
    try:
        symbol = mapping[symbol]
        return symbol
    except KeyError as ke:
        res = mg.query('symbol:%s' % symbol, size=1, fields='uniprot')['hits']
        if res and 'uniprot' in res[0]:
            if 'Swiss-Prot' in res[0]['uniprot']:
                return res[0]['uniprot']['Swiss-Prot']

        print("Couldn't find %s" % symbol)
        return None



def to_interactions(df: pd.DataFrame, mg, subsample=1.0):
    genes = set()
    np.random.seed(5005)

    mapping = {}
    for _, row in df[df['INTERACTION_TYPE'].str.contains('ProteinReference')].iterrows():
        uniprot_id = re.findall(r'uniprot knowledgebase:(\S+)', row['INTERACTION_DATA_SOURCE'])[0]
        hgnc_name = row['PARTICIPANT_A']
        assert hgnc_name not in mapping
        mapping[hgnc_name] = uniprot_id


    df = df[df['INTERACTION_TYPE'].isin(INTERACTION_TYPES)]
    df = df.fillna({"INTERACTION_PUBMED_ID": ""})
    genes.update(id_ for id_ in df['PARTICIPANT_A'].unique() if not ':' in id_)
    genes.update(id_ for id_ in df['PARTICIPANT_B'].unique() if not ':' in id_)

    if subsample < 1.0:
        genes = subsample_genes(df, size=int(len(genes)*subsample))

    genes = sorted(genes)


    interactions = defaultdict(set)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        head = row['PARTICIPANT_A']
        tail = row['PARTICIPANT_B']

        head = hgnc_to_uniprot(head, mapping, mg)
        tail = hgnc_to_uniprot(tail, mapping, mg)

        if not head or not tail:
            continue


        relation_type = row['INTERACTION_TYPE']
        pmids = row['INTERACTION_PUBMED_ID'].split(';')

        relation = f"{head},{relation_type},{tail}"
        interactions[relation].update(pmids)

    result = {k: list(v) for k, v in interactions.items()}

    return result


def split(interactions):
    """
    Split `interactions` into train/dev/test sets
    Entailed interactions are removed and added in a later step to make sure that the entailing and entailed relations are in the same fold
    """
    entities = set()
    for k in interactions:
        e1, r, e2 = k.split(',')
        entities.add(e1)
        entities.add(e2)

    filtered_interactions = {}
    blacklist = set() # blacklist is used to filter entailed relations, e.g. for `A controls-phosphorylation-of B` `A controls-state-change-of B` is blacklisted
    for k in interactions:
        if k in blacklist:
            continue

        triple = k.split(",")


        if triple[1] in {"controls-phosphorylation-of", "controls-transport-of"}:
            blacklist.add(",".join([triple[0], "controls-state-change-of", triple[2]]))

        if triple[1] == "in-complex-with":
            blacklist.add(",".join([triple[2], "in-complex-with", triple[0]]))

    for k in interactions:
        triple = k.split(",")

        if k not in blacklist:
            filtered_interactions[k] = interactions[k]

    keys = sorted(filtered_interactions.keys())
    np.random.seed(5005)
    np.random.shuffle(keys)
    idx1 = int(len(keys) * 0.6)
    idx2 = idx1 + int(len(keys) * 0.1)

    train_interactions = augment_interactions({k: filtered_interactions[k] for k in keys[:idx1]})
    dev_interactions = augment_interactions({k: filtered_interactions[k] for k in keys[idx1:idx2]})
    test_interactions = augment_interactions({k: filtered_interactions[k] for k in keys[idx2:]})


    return train_interactions, dev_interactions, test_interactions, filtered_interactions

def add_na_interactions(interactions, factor=10):
    """
    Add NA interaction if (e1, r, e2) and (e2, r, e1) are both not in interactions
    NA interactions are randomly sampled and the number is determined by `factor`*len(interactions)
    Assumes that inverse relations are added later on
    """
    pairs = set()
    entities = set()
    na_interactions = set()
    for interaction in interactions:
        e1, _, e2 = interaction.split(',')
        pairs.add(tuple(sorted([e1, e2])))
        entities.update([e1, e2])
    
    for pair in itertools.combinations(entities, 2):
        e1, e2 = sorted(pair) 
        if (e1, e2) not in pairs:
            na_interactions.add(",".join([e1, "NA", e2]))
    
    try:
        na_interactions = np.random.choice(list(na_interactions),
            size=int(factor*len(interactions)), replace=False)
    except ValueError:
        pass
    
    for interaction in na_interactions:
        interactions[interaction] = []
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()

    mg = mygene.MyGeneInfo()
    mg.set_caching('mg_cache')

    df = pd.read_csv(args.input, sep='\t', header=0)
    subsample = 0.01 if args.small else 1.0
    interactions = to_interactions(df, mg, subsample=subsample)
    fname = args.input
    if args.small:
        fname += '_small'
    with open(fname + '.train.json', 'w') as f_train, open(fname + '.dev.json', 'w') as f_dev, \
        open(fname + '.test.json', 'w') as f_test, open(fname + '.json', 'w') as f_all:

        add_na_interactions(interactions)
        train, dev, test, interactions = split(interactions)

        json.dump(train, f_train)
        json.dump(dev, f_dev)
        json.dump(test, f_test)

        json.dump(interactions, f_all)

