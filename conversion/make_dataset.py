import argparse
import json
from collections import defaultdict

import pandas as pd
import mygene
import numpy as np

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

def to_interactions(df: pd.DataFrame, mg):
    genes = set()
    df = df[df['INTERACTION_TYPE'].isin(INTERACTION_TYPES)]
    df = df.fillna({"INTERACTION_PUBMED_ID": ""})
    genes.update(id_ for id_ in df['PARTICIPANT_A'].unique() if not ':' in id_)
    genes.update(id_ for id_ in df['PARTICIPANT_B'].unique() if not ':' in id_)
    genes = sorted(genes)

    mapping = {g: None for g in genes}
    scores = {g: np.float('-inf') for g in genes}

    mg_result = mg.querymany(genes, scopes='symbol', fields='entrezgene', species='human')

    for res in mg_result:
        if '_score' not in res:
            continue

        if res['_score'] > scores[res['query']]:
            scores[res['query']] = res['_score']
            mapping[res['query']] = res['_id']

    interactions = defaultdict(set)
    for _, row in df.iterrows():
        head = row['PARTICIPANT_A']
        tail = row['PARTICIPANT_B']
        if head not in mapping or tail not in mapping:
            continue



        head = mapping[head]
        tail = mapping[tail]
        relation_type = row['INTERACTION_TYPE']
        pmids = row['INTERACTION_PUBMED_ID'].split(';')

        relation = f"{head},{relation_type},{tail}"
        interactions[relation].update(pmids)

    result = {k: list(v) for k, v in interactions.items()}

    return result


def split(interactions):
    filtered_interactions = {}
    blacklist = set()
    for k in interactions:
        if k in blacklist:
            continue

        triple = k.split(",")
        if triple[1] in {"controls-phosphorylation-of", "controls-transport-of"}:
            blacklist.add(",".join([triple[0], "controls-state-change-of", triple[2]]))

        if triple[1] == "in-complex-with":
            blacklist.add(",".join([triple[2], "in-complex-with", triple[0]]))
    for k in interactions:
        if k not in blacklist:
            filtered_interactions[k] = interactions[k]

    interactions = filtered_interactions
    keys = list(interactions.keys())
    np.random.seed(5005)
    np.random.shuffle(keys)
    idx1 = int(len(keys) * 0.6)
    idx2 = idx1 + int(len(keys) * 0.1)

    train_interactions = augment_interactions({k: interactions[k] for k in keys[:idx1]})
    dev_interactions = augment_interactions({k: interactions[k] for k in keys[idx1:idx2]})
    test_interactions = augment_interactions({k: interactions[k] for k in keys[idx2:]})

    return train_interactions, dev_interactions, test_interactions, filtered_interactions



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    mg = mygene.MyGeneInfo()

    df = pd.read_csv(args.input, sep='\t', header=0)
    interactions = to_interactions(df, mg)
    with open(args.input + '.train.json', 'w') as f_train, open(args.input + '.dev.json', 'w') as f_dev, \
        open(args.input + '.test.json', 'w') as f_test, open(args.input + '.json', 'w') as f_all:
        train, dev, test, interactions = split(interactions)

        json.dump(train, f_train)
        json.dump(dev, f_dev)
        json.dump(test, f_test)

        json.dump(interactions, f_all)

