import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True, type=Path)
    parser.add_argument('--preds', required=True, type=Path)

    args = parser.parse_args()


    with args.anns.open() as f:
        anns = {k: v for k, v in json.load(f).items() if v['mentions']}

    y_true = []
    y_score = []
    prov_aps = []
    prov_random_aps = []
    n_snippets = 0
    n_pos_snippets = 0
    predicted_pairs = set()
    with args.preds.open() as f:
        lines = f.readlines()
    for line in tqdm(lines):
        pred = json.loads(line.strip())
        predicted_pairs.add(','.join(pred['entities']))
        ann = anns[','.join(pred['entities'])]
        for label, score in pred['labels']:
            y_score.append(score)
            y_true.append(label in ann['relations'])

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    sns.set()

    plt.plot(precision, recall)
    plt.savefig(str(args.preds) + '_pr.jpg')





