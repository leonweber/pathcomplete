import argparse
import json

from pathlib import Path

from util import TAX_IDS, load_homologene_uf, load_homologene


def reshuffle(train, dev, test, homologene):
    all_pairs = set(train) | set(dev) | set(test)
    train_prop = len(train) / len(all_pairs)
    dev_prop = len(dev) / len(all_pairs)
    test_prop = len(test) / len(all_pairs)

    pair_clusters = set()

    for pair in all_pairs:
        e1, e2 = pair.split(",")
        cluster1 = homologene[e1]
        cluster2 = homologene[e2]
        pair_clusters.add((cluster1, cluster2))
    print(len(pair_clusters), len(all_pairs))










if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)

    args = parser.parse_args()

    with args.train.open() as f:
        train = json.load(f)
    with args.dev.open() as f:
        dev = json.load(f)
    with args.test.open() as f:
        test = json.load(f)
    with open('data/geneid2uniprot.json') as f:
        geneid2uniprot = json.load(f)

    homologene = load_homologene_uf(species={TAX_IDS['rat'], TAX_IDS['mouse'], TAX_IDS['rabbit'], TAX_IDS['hamster'], TAX_IDS['human']},
                                    gene_conversion=geneid2uniprot)

    train, dev, test = reshuffle(train, dev, test, homologene)

    with (args.train.parent/'train_reshuffled.json', 'w') as f:
        json.dump(train, f)
    with (args.dev.parent/'dev_reshuffled.json', 'w') as f:
        json.dump(dev, f)
    with (args.test.parent/'test_reshuffled.json', 'w') as f:
        json.dump(test, f)


