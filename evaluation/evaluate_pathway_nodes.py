import argparse
import json
import numpy as np

from pathlib import Path

from sklearn.metrics import average_precision_score

from conversion.util import slugify

def f1_nodes(true_nodes, pred_nodes):
    true_nodes = set(true_nodes)
    pred_nodes = set(pred_nodes)

    tps = 0
    fps = 0
    fns = 0

    tps += len(true_nodes & pred_nodes)
    fps += len(pred_nodes - true_nodes)
    fns += len(true_nodes - pred_nodes)

    try:
        p = tps / (tps + fps)
        r = tps / (tps + fns)
        f1 = 2 * p * r / (p + r)
    except:
        p = r = f1 = 0.0
    return p, r, f1

def evaluate_ranking(anns, preds, ignore_fns=False):
    aps = []
    for pw_name, pred_nodes in preds.items():
        y_true = []
        true_nodes = set(anns[pw_name])
        for node in pred_nodes:
            y_true.append(node in true_nodes)
        y_score = 1/np.arange(1, len(y_true)+1)

        aps.append(average_precision_score(y_true, y_score))


    if not ignore_fns:
        for ann in anns:
            if ann not in preds:
                aps.append(0.0)

    return np.mean(aps)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True, type=Path)
    parser.add_argument('--preds', required=True, type=Path)
    parser.add_argument('--slugify', action='store_true')
    parser.add_argument('--ignore_fns', action='store_true')

    args = parser.parse_args()

    with args.anns.open() as f:
        anns = json.load(f)
    if args.slugify:
        new_anns = {}
        for name, ann in anns.items():
            new_anns[slugify(name)] = ann
        anns = new_anns

    with args.preds.open() as f:
        preds = json.load(f)
    if args.slugify:
        new_preds = {}
        for name, ann in preds.items():
            new_preds[slugify(name)] = ann
        preds = new_preds

    print(evaluate_ranking(anns, preds, ignore_fns=args.ignore_fns))
