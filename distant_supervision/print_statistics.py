import argparse
import json
from pathlib import Path
import seaborn as sns
import numpy as np



def print_statistics(data):
    n_pos = 0
    n_neg = 0
    n_pos_mentions = []
    n_neg_mentions = []
    n_rels = []


    for k, v in data.items():
        try:
            if not v['mentions']:
                continue

            n_rels.append(len(v['relations']))
            if v['relations'] and v['relations'][0] != 'NA':
                n_pos_mentions.append(len(v['mentions']))
                n_pos += 1
            else:
                n_neg_mentions.append(len(v['mentions']))
                n_neg += 1
        except KeyError:
            n_neg_mentions.append(len(v['mentions']))
            n_neg += 1

    print("Positive pairs:", n_pos)
    print("Negative pairs:", n_neg)
    # sns.set()
    # pos_mention_plot = sns.boxplot(n_pos_mentions).get_figure()
    # neg_mention_plot = sns.boxplot(n_neg_mentions).get_figure()
    # pos_mention_plot.savefig('pos_mention_boxplot.png')
    # neg_mention_plot.savefig('neg_mention_boxplot.png')
    print(f"Positive mentions: {np.mean(n_pos_mentions)} +/- {np.std(n_pos_mentions)}")
    print(f"Negative mentions: {np.mean(n_neg_mentions)} +/- {np.std(n_neg_mentions)}")
    print(f"Relations: {np.mean(n_rels)} +/- {np.std(n_rels)}")







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    args = parser.parse_args()

    all_data = {}
    for d in args.data:
        with d.open() as f:
            all_data.update(json.load(f))

    print_statistics(all_data)
