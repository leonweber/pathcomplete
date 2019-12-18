import argparse
from collections import defaultdict

import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_curve


def dhyper(desired, white, black, draws):
    # prob. to draw `desired` white balls from `white` + `black` balls in with `draws`
    return stats.hypergeom.pmf(desired, white + black, white, draws)


def compute_random_ap(n_relevant, n_total):
    # expected AP of random baseline
    # taken from https://ufal.mff.cuni.cz/pbml/103/art-bestgen.pdf
    # for large populations proportion of relevant elements is a good estimate (delta < 0.01)
    ap = 0
    for i in range(1, n_relevant + 1):
        for n in range(i, n_total - n_relevant + i + 1):
            ap += dhyper(i, n_relevant, n_total - n_relevant, n) * (i / n) * (i / n)
    ap /= n_relevant
    return ap


def evaluate_random_baseline(anns, preds):
    prov_random_aps = []
    y_true = []
    for line in tqdm(preds):
        pred = json.loads(line.strip())
        try:
            ann = anns[','.join(pred['entities'])]
        except KeyError:
            continue
        y_prov = [m[1] == 'direct' for m in ann['mentions']]

        for label, score in pred['labels']:
            y_true.append(label in ann['relations'])

        if sum(y_prov):
            try:
                prov_aps.append(average_precision_score(y_prov, pred['alphas']))
            except (ValueError, KeyError):
                prov_aps.append(0)
            prov_random_aps.append(compute_random_ap(sum(y_prov), len(y_prov)))

    return np.mean(y_true), prov_random_aps


def evaluate(anns, preds):
    y_true = []
    y_score = []

    y_true_by_relation = defaultdict(list)
    y_score_by_relation = defaultdict(list)

    prov_aps = []
    n_snippets = 0
    n_pos_snippets = 0
    predicted_pairs = set()
    for line in tqdm(preds):
        pred = json.loads(line.strip())
        try:
            ann = anns[','.join(pred['entities'])]
        except KeyError:
            continue
        predicted_pairs.add(','.join(pred['entities']))
        for label, score in pred['labels']:
            y_score.append(score)
            y_true.append(label in ann['relations'])
            y_score_by_relation[label].append(score)
            y_true_by_relation[label].append(label in ann['relations'])
        y_prov = [m[1] == 'direct' for m in ann['mentions']]
        if sum(y_prov):
            try:
                prov_aps.append(average_precision_score(y_prov, pred['alphas']))
            except (ValueError, KeyError):
                prov_aps.append(0)
            n_snippets += len(y_prov)
            n_pos_snippets += sum(y_prov)

    assert predicted_pairs.issubset(set(anns))
    max_recall = len(predicted_pairs) / len(anns)
    ap = average_precision_score(y_true, y_score, average='micro') * max_recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    thresholds = np.array([0] + thresholds.tolist())
    recall = recall * max_recall

    df_all = pd.DataFrame({
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    })
    df_all['relation'] = 'all'

    for rel in y_true_by_relation:
        y_true = y_true_by_relation[rel]
        y_score = y_score_by_relation[rel]
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        recall = recall * max_recall
        thresholds = np.array([0] + thresholds.tolist())

        df = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds
        })
        df['relation'] = rel
        df_all = pd.concat([df_all, df])

    return ap, prov_aps, df_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True, type=Path)
    parser.add_argument('--preds', required=True, type=Path)

    args = parser.parse_args()

    with args.anns.open() as f:
        anns = {k: v for k, v in json.load(f).items() if v['mentions']}

    with args.preds.open() as f:
        preds = f.readlines()

    ap, prov_aps, df = evaluate(anns, preds)
    random_ap, random_prov_aps = evaluate_random_baseline(anns, preds)

    print('rel AP:', ap, '(random baseline:', random_ap, ')')
    print('prov mAP:', np.mean(prov_aps), '(random baseline:', np.mean(random_prov_aps), ')')
