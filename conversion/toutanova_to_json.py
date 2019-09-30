import json

import mygene
import argparse
from pathlib import Path
from tqdm import tqdm

def hgnc_to_uniprot(symbol, mapping, mg):
    try:
        symbol = mapping[symbol]
        return symbol
    except KeyError as ke:
        res = mg.query('symbol:%s' % symbol, size=1, fields='uniprot')['hits']
        if res and 'uniprot' in res[0]:
            if 'Swiss-Prot' in res[0]['uniprot']:
                uniprot = res[0]['uniprot']['Swiss-Prot']
                return [uniprot]

        print("Couldn't find %s" % symbol)
        return None

def transform_data(data, mg):
    transformed_data = {}
    n_failures = 0
    for triple in tqdm(data):
        e1, r, e2 = triple.split('\t')[:3]

        try:
            e1_hgnc = hgnc_to_uniprot(e1, {}, mg)[0]
            e2_hgnc = hgnc_to_uniprot(e2, {}, mg)[0]
        except TypeError:
            n_failures += 1
            continue

        e1 = e1_hgnc or e1
        e2 = e1_hgnc or e2

        transformed_data[f"{e1},{r},{e2}"] = []

    if n_failures > 0:
        print(f"WARNING: Could not transform {n_failures} triples.")

    return transformed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path)
    parser.add_argument('--dev', type=Path)
    parser.add_argument('--test', type=Path)
    args = parser.parse_args()

    mg = mygene.MyGeneInfo()
    mg.set_caching('mg_cache')

    with args.train.open() as f, (args.train.parent/'train.json').open('w') as f_out:
        train = [l.strip() for l in f]
        transformed_train = transform_data(train, mg)
        json.dump(transformed_train, f_out)

    with args.dev.open() as f, (args.dev.parent / 'dev.json').open('w') as f_out:
        dev = [l.strip() for l in f]
        transformed_dev = transform_data(dev, mg)
        json.dump(transformed_dev, f_out)

    with args.test.open() as f, (args.test.parent / 'test.json').open('w') as f_out:
        test = [l.strip() for l in f]
        transformed_test = transform_data(test, mg)
        json.dump(transformed_test, f_out)

