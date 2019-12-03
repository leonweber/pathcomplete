import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from scipy import stats
from sklearn.metrics import average_precision_score


def dhyper(desired, white, black, draws):
    # prob. to draw `desired` white balls from `white` + `black` balls in with `draws`
    return stats.hypergeom.pmf(desired, white+black, white, draws)


def random_ap(n_relevant, n_total):
    # expected AP of random baseline
    # taken from https://ufal.mff.cuni.cz/pbml/103/art-bestgen.pdf
    # for large populations proportion of relevant elements is a good estimate (delta < 0.01)
    ap = 0
    for i in range(1, n_relevant+1):
        for n in range(i, n_total-n_relevant+i+1):
            ap += dhyper(i, n_relevant, n_total-n_relevant, n) * (i/n) * (i/n)
    ap /= n_relevant
    return ap




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
        y_prov = [m[1] == 'direct' for m in ann['mentions']]
        if sum(y_prov):
            try:
                prov_aps.append(average_precision_score(y_prov, pred['alphas']))
            except ValueError:
                prov_aps.append(0)
            prov_random_aps.append(random_ap(sum(y_prov), len(y_prov)))
            n_snippets += len(y_prov)
            n_pos_snippets += sum(y_prov)

    assert predicted_pairs.issubset(set(anns))
    max_recall = len(predicted_pairs)/len(anns)
    ap = average_precision_score(y_true, y_score, average='micro') * max_recall

    print('max recall:', max_recall)
    print('rel AP:', ap, '(random baseline:', np.mean(y_true), ')')
    print('prov mAP:', np.mean(prov_aps), '( N:', n_snippets, ', random baseline:', np.mean(prov_random_aps), ')')



