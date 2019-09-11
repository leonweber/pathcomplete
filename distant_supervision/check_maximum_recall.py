import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--ds', required=True, type=Path)

    args = parser.parse_args()

    ds_pairs = set()
    with args.ds.open() as f:
        for line in f:
            fields = line.split('\t')
            if fields[4] != 'NA':
                ds_pairs.add(tuple(sorted(fields[:2])))

    data_pairs = set()
    with args.data.open() as f:
        data = json.load(f)
        for k in data:
            e1, r, e2 = k.split(',')
            if r != 'NA':
                data_pairs.add(tuple(sorted([e1, e2])))

    print(len(ds_pairs & data_pairs) / len(data_pairs))

