import argparse
import itertools
import json
from collections import defaultdict
from copy import deepcopy

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


def filter_triples(triples, entities):
    filtered_triples = set()
    for triple in triples:
        if triple.split(",")[1] == "NA":
            continue
        e1 = triple.split(",")[0]
        e2 = triple.split(",")[2]

        if e1 in entities and e2 in entities:
            filtered_triples.add(triple)

    return filtered_triples


def evaluate_relations(anns, preds, entities=None, baseline=None, has_scores=True):

    true_relations = set(anns)

    if baseline:
        baseline_prec = evaluate_relations(anns, baseline, entities, has_scores=False)['prec']
        pred_relations = list(reversed(sorted(preds, key=lambda x: float(preds[x]["score"]))))
        precs = []
        for i, _ in enumerate(pred_relations, start=1):
            current_preds = set(pred_relations[:i])
            if entities:
                current_preds = filter_triples(current_preds, entities)
            tps = current_preds & true_relations
            precs.append(len(tps) / i)
        precs = np.array(precs)
        try:
            i = np.where(precs >= baseline_prec)[0][-1]
            pred_relations = set(pred_relations[:i])
        except IndexError:
            pred_relations = set()
    else:
        pred_relations = set(p for p, meta in preds.items() if "score" not in meta or meta["score"] >= 0.6)

    if has_scores:
        proba_scores = [preds[rel]["score"] for rel in preds] + [0.0] * len(true_relations - set(preds))
        y_true = [int(rel in true_relations) for rel in preds] + [1] * len(true_relations - set(preds))

        prec_vals, rec_vals, _ = precision_recall_curve(y_true, proba_scores)
        prec_vals = prec_vals.tolist()
        rec_vals = rec_vals.tolist()
    else:
        prec_vals = None
        rec_vals = None


    if baseline and preds:
        y_true = [int(rel in true_relations) for rel in preds if rel not in baseline]
        proba_scores = [preds[rel]["score"] for rel in preds if rel not in baseline]

        new_prec_vals, new_rec_vals, _ = precision_recall_curve(y_true, proba_scores)
        new_prec_vals = new_prec_vals.tolist()
        new_rec_vals = new_rec_vals.tolist()
    else:
        new_prec_vals = None
        new_rec_vals = None

    if entities:
        pred_relations = filter_triples(pred_relations, entities)
        true_relations = filter_triples(true_relations, entities)

    tps = true_relations & pred_relations
    fps = pred_relations - true_relations
    fns = true_relations - pred_relations

    if baseline:
        tps_new = len(tps - set(baseline))
    else:
        tps_new = None

    if len(tps) == 0:
        return {
        'prec': 0,
        'rec': 0,
        'f1': 0,
        'support': len(true_relations),
        'new': 0,
        'prec_vals': prec_vals,
        'rec_vals': rec_vals,
        'new_prec_vals': new_prec_vals,
        'new_rec_vals': new_rec_vals,
    }

    prec = len(tps) / (len(tps) + len(fps))
    rec = len(tps) / (len(tps) + len(fns))

    if prec == 0 or rec == 0:
        return {
            'prec': 0,
            'rec': 0,
            'f1': 0,
            'support': len(true_relations),
            'new': 0,
            'prec_vals': prec_vals,
            'rec_vals': rec_vals,
            'new_prec_vals': new_prec_vals,
            'new_rec_vals': new_rec_vals,
        }

    f1 = 2 * prec * rec / (prec + rec)

    result = {
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'support': len(true_relations),
        'new': tps_new,
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
    preds = augment_preds(preds)

    entities = set()
    if args.pathway:
        with open(args.pathway) as f:
            pw_triples = json.load(f)
            for triple in pw_triples:
                entities.add(triple.split(',')[0])
                entities.add(triple.split(',')[2])
    else:
        entities = None


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

        res = evaluate_relations(rel_anns, rel_preds, entities, baseline, has_scores=not args.no_scores)

        new_str = f", New: {res['new']}" if res['new'] else ""
        print(f"{relation}: {res['prec']}/{res['rec']}/{res['f1']}{new_str}, Support: {res['support']}")
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


