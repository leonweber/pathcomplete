import argparse
import json
import os

import numpy as np
from collections import defaultdict

from pathlib import Path

from sklearn.model_selection import train_test_split


def combine_data(input, split):
    combined_data = {}
    for i in input:
        i = Path(i)
        with open(i/f'{split}.json') as f:
            data = json.load(f)

        for k, v in data.items():
            if k not in combined_data:
                combined_data[k] = v
            else:
                combined_mentions = [tuple(i) for i in combined_data[k]['mentions'] + v['mentions']]
                combined_data[k]['mentions'] = list(set(combined_mentions))
                combined_data[k]['relations'] = list(set(combined_data[k]['relations'] + v['relations']))

    return combined_data


def downsample_bag_size(data, max_bag_size):
    for v in data.values():
        direct_mentions = [m for m in v['mentions'] if m[1] == 'direct']
        indirect_mentions = sorted([m for m in v['mentions'] if m[1] != 'direct'], key=lambda x: x[0])
        assert len(direct_mentions) < max_bag_size
        n_left = max_bag_size - len(direct_mentions)

        if len(indirect_mentions) > n_left:
            np.random.seed(5005)
            arr = np.empty(len(indirect_mentions), dtype=object)
            arr[:] = indirect_mentions[:]
            indirect_mentions = arr
            indirect_mentions = np.random.choice(indirect_mentions, n_left, replace=False).tolist()
        v['mentions'] = direct_mentions + indirect_mentions


def downsample_negative_pairs(data):
    pairs = set(data)
    positive_pairs = set(k for k,v in data.items() if v['relations'])
    negative_pairs = pairs - positive_pairs
    n_negative = len(positive_pairs)*10

    np.random.seed(5005)
    sampled_pairs = set(np.random.choice(sorted(negative_pairs), n_negative, replace=False).tolist())

    for pair in negative_pairs - sampled_pairs:
        del data[pair]


def train_dev_split(data):
    pairs = list(data)
    train_pairs, dev_pairs = train_test_split(pairs, train_size=0.8)

    train_data = {p: data[p] for p in train_pairs}
    dev_data = {p: data[p] for p in dev_pairs}

    return train_data, dev_data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    parser.add_argument('output', type=Path)
    parser.add_argument('--max_bag_size', type=int, default=100)

    args = parser.parse_args()

    combined_train_data = combine_data(args.input, split='train')
    combined_test_data = combine_data(args.input, split='dev')
    downsample_bag_size(combined_train_data, args.max_bag_size)
    downsample_negative_pairs(combined_train_data)
    downsample_bag_size(combined_test_data, args.max_bag_size)
    downsample_negative_pairs(combined_test_data)

    os.makedirs(args.output, exist_ok=True)
    train_data, dev_data = train_dev_split(combined_train_data)
    with (args.output / 'train.json').open('w') as f:
        json.dump(train_data, f)
    with (args.output / 'dev.json').open('w') as f:
        json.dump(dev_data, f)
    with (args.output / 'test.json').open('w') as f:
        json.dump(combined_test_data, f)
