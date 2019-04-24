import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    args = parser.parse_args()

    os.makedirs(f'data/link_prediction/{args.name}', exist_ok=True)

    with open(f'data/PathwayCommons11.{args.name}.hgnc.txt.train.json') as f:
        train = json.load(f)
        train_triples = [t.replace(',', '\t') for t in train]
    with open(f"data/link_prediction/{args.name}/train.txt", 'w') as f:
        f.write("\n".join(train_triples))

    with open(f'data/PathwayCommons11.{args.name}.hgnc.txt.dev.json') as f:
        dev = json.load(f)
        dev_triples = [t.replace(',', '\t') for t in dev]
    with open(f"data/link_prediction/{args.name}/valid.txt", 'w') as f:
        f.write("\n".join(dev_triples))

    with open(f'data/PathwayCommons11.{args.name}.hgnc.txt.test.json') as f:
        test = json.load(f)
        test_triples = [t.replace(',', '\t') for t in test]
    with open(f"data/link_prediction/{args.name}/test.txt", 'w') as f:
        f.write("\n".join(test_triples))
