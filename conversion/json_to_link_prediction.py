import argparse
import json
import os
from utils import datadir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    os.makedirs(f'{datadir}/link_prediction/{args.name}', exist_ok=True)

    with open(f'{datadir}/PathwayCommons11.{args.name}.hgnc.txt.train.json') as f:
        train = json.load(f)
        train_triples = [t.replace(',', '\t') for t in train]
    with open(f"{datadir}/link_prediction/{args.name}/train.txt", 'w') as f:
        f.write("\n".join(train_triples))

    with open(f'{datadir}/PathwayCommons11.{args.name}.hgnc.txt.dev.json') as f:
        dev = json.load(f)
        dev_triples = [t.replace(',', '\t') for t in dev]
    with open(f"{datadir}/link_prediction/{args.name}/valid.txt", 'w') as f:
        f.write("\n".join(dev_triples))

    with open(f'{datadir}/PathwayCommons11.{args.name}.hgnc.txt.test.json') as f:
        test = json.load(f)
        test_triples = [t.replace(',', '\t') for t in test]
    with open(f"{datadir}/link_prediction/{args.name}/test.txt", 'w') as f:
        f.write("\n".join(test_triples))

    entities = set()
    relations = set()
    for t in train_triples + dev_triples + test_triples:
        t = t.split('\t')
        entities.update([t[0], t[2]])
        relations.add(t[1])
    entities = sorted(entities)
    relations = sorted(relations)

    with open(f"{datadir}/link_prediction/{args.name}/entities.dict", 'w') as f:
        for i, entity in enumerate(entities):
            f.write(f"{i}\t{entity}\n")
    with open(f"{datadir}/link_prediction/{args.name}/relations.dict", 'w') as f:
        for i, relation in enumerate(relations):
            f.write(f"{i}\t{relation}\n")

