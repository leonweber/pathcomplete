import argparse
import itertools
import json
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.metrics.ranking import precision_recall_curve


def augment_preds(preds):
    augmented_preds = dict(preds)
    for pred in preds:
        e1, r, e2 = pred.split(',')
        if r == 'controls-phosphorylation-of':
            augmented_preds[f"{e1},controls-state-change-of,{e2}"] = preds[pred]
        # if r == 'controls-transport-of':
        #     augmented_preds[f"{e1},controls-state-change-of,{e2}"] = preds[pred]
        if r == 'in-complex-with':
            augmented_preds[f"{e2},in-complex-with,{e1}"] = preds[pred]
    
    return augmented_preds




def evaluate_relations(anns, preds, baseline=None):

    true_relations = set(anns)

    proba_scores = [preds[rel]["score"] for rel in preds]
    y_true = [int(rel in true_relations) for rel in preds]

    proba_scores = proba_scores + [min(proba_scores)-1] * len(true_relations - set(preds))
    y_true = y_true + [1] * len(true_relations - set(preds))

    prec_vals, rec_vals, _ = precision_recall_curve(y_true, proba_scores)
    ap = np.sum(np.diff(np.insert(rec_vals[::-1], 0, 0)) * prec_vals[::-1])

    prec_vals = prec_vals.tolist()
    rec_vals = rec_vals.tolist()


    if baseline and preds:
        y_true = [int(rel in true_relations) for rel in preds if rel not in baseline]
        proba_scores = [preds[rel]["score"] for rel in preds if rel not in baseline]

        new_prec_vals, new_rec_vals, _ = precision_recall_curve(y_true, proba_scores)
        new_ap = np.sum(np.diff(np.insert(new_rec_vals[::-1], 0, 0)) * new_prec_vals[::-1])

        new_prec_vals = new_prec_vals.tolist()
        new_rec_vals = new_rec_vals.tolist()
    else:
        new_prec_vals = None
        new_rec_vals = None
        new_ap = None


    result = {
        'support': len(true_relations),
        'ap': ap,
        'new_ap': new_ap,
        'prec_vals': prec_vals,
        'rec_vals': rec_vals,
        'new_prec_vals': new_prec_vals,
        'new_rec_vals': new_rec_vals,
    }

    return result


def evaluate_pairs(anns, preds, entities=None):
    if entities:
        anns = filter_triples(anns, entities)
        preds = filter_triples(preds, entities)

    true_pairs = set((t.split(',')[0], t.split(',')[2]) for t in anns)
    pred_pairs = set((t.split(',')[0], t.split(',')[2]) for t in preds)

    tps = len(true_pairs & pred_pairs)
    fps = len(pred_pairs - true_pairs)
    fns = len(true_pairs - pred_pairs)

    if tps == 0:
        return 0, 0, 0, len(true_pairs)

    prec = tps / (tps + fps)
    rec = tps / (tps + fns)

    if prec == 0 or rec == 0:
        return 0, 0, 0, len(true_pairs)

    f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1, len(true_pairs)


def evaluate_provenance(anns, preds):
    true_predictions = set(anns.keys()) & set(preds.keys())

    tps = 0
    fps = 0
    fns = 0
    support = 0


    for pred in true_predictions:
        pred_pmids = set(p for p, _ in preds[pred])
        true_pmids = set(anns[pred])

        tps += len(pred_pmids & true_pmids)
        fps += len(pred_pmids - true_pmids)
        fns += len(true_pmids - pred_pmids)
        support += len(true_pmids)

    if not true_predictions:
        return 0, 0, 0, support


    prec = tps / (tps + fps)
    rec = tps / (tps + fns)

    if prec == 0 or rec == 0:
        return 0, 0, 0, support

    f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1, support


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True)
    parser.add_argument('--preds', required=True)
    parser.add_argument('--pathway', required=False)
    parser.add_argument('--baseline', required=False)
    parser.add_argument('--filter', required=False)
    parser.add_argument('--no_scores', action='store_true')
    parser.add_argument('--out', required=False)

    args = parser.parse_args()

    with open(args.anns) as f:
        anns = json.load(f)
        anns = {k: v for k, v in anns.items() if k.split(',')[1] != 'NA'}
        if args.filter:
            with open(args.filter) as f:
                filter = json.load(f)
            anns = {k: v for k, v in anns.items() if k not in filter}

    with open(args.preds) as f:
        preds = json.load(f)
        if args.filter:
            with open(args.filter) as f:
                filter = json.load(f)
            preds = {k: v for k, v in preds.items() if k not in filter}

    entities = set()
    if args.pathway:
        with open(args.pathway) as f:
            pw_triples = json.load(f)
            for triple in pw_triples:
                entities.add(triple.split(',')[0])
                entities.add(triple.split(',')[2])

        anns = {k: v for k, v in anns.items() if k.split(',')[0] in entities and k.split(',')[2] in entities}
        preds = {k: v for k, v in preds.items() if k.split(',')[0] in entities and k.split(',')[2] in entities}

    preds = augment_preds(preds)

    df = defaultdict(list)

    if args.baseline:
        with open(args.baseline) as f:
            baseline  = json.load(f)
        baseline = {k: v for k, v in baseline.items() if k.split(',')[1] != 'NA'}
        baseline = augment_preds(baseline)
    else:
        baseline = None

    relations = set(t.split(',')[1] for t in anns)
    for relation in sorted(itertools.chain(['all'], relations)):
        if relation == "NA":
            continue

        if relation != 'all':
            rel_anns = {k: v for k, v in anns.items() if k.split(',')[1] == relation}
            rel_preds = {k: v for k, v in preds.items() if k.split(',')[1] == relation}
        else:
            rel_anns = deepcopy(anns)
            rel_preds = deepcopy(preds)

        res = evaluate_relations(rel_anns, rel_preds, baseline)

        print(relation)
        pprint({k: v for k,v in res.items() if 'vals' not in k})
        df['rel'].append(relation)

        for r, value in res.items():
            if isinstance(value, list):
                new_value = " ".join([str(v) for v in value])
                df[r].append(new_value)
            else:
                df[r].append(value)

    df['data'] = [args.anns] * len(df['prec'])
    df['predictor'] = [args.preds] * len(df['prec'])

    if args.out:
        df = pd.DataFrame(df)
        df.to_csv(args.out, sep='\t')


