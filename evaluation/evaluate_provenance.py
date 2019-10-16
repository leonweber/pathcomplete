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


def evaluate_provenance(anns, preds):
    tps = set(anns) & set(preds)

    proba_scores = []
    y_true = []
    n_true_pmids = 0

    for tp in tps:
        true_provenance = anns[tp]
        n_true_pmids += len(true_provenance)
        for pmid, score in preds[tp]['provenance'].items():
            proba_scores.append(score)
            y_true.append(pmid in true_provenance)

    if n_true_pmids:
        max_recall = sum(y_true) / n_true_pmids
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, proba_scores)
        rec_vals = rec_vals * max_recall
        ap = np.sum(np.diff(np.insert(rec_vals[::-1], 0, 0)) * prec_vals[::-1])
    else:
        ap = np.nan

    return ap


def evaluate_relations(anns, preds, baseline=None):

    true_relations = set(anns)

    proba_scores = [preds[rel]["score"] for rel in preds]
    y_true = [int(rel in true_relations) for rel in preds]

    max_recall = len(true_relations & set(preds)) / len(true_relations)

    prec_vals, rec_vals, _ = precision_recall_curve(y_true, proba_scores)
    rec_vals = rec_vals * max_recall
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

    provenance_ap = evaluate_provenance(anns, preds)


    result = {
        'support': len(true_relations),
        'ap': ap,
        'provenance_ap': provenance_ap,
        'new_ap': new_ap,
        'prec_vals': prec_vals,
        'rec_vals': rec_vals,
        'new_prec_vals': new_prec_vals,
        'new_rec_vals': new_rec_vals,
    }

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True)
    parser.add_argument('--preds', required=True)
    parser.add_argument('--pathway', required=False)
    parser.add_argument('--baseline', required=False)
    parser.add_argument('--filter', required=False)
    parser.add_argument('--rel_blacklist', required=False)
    parser.add_argument('--no_scores', action='store_true')
    parser.add_argument('--augment_preds', action='store_true')
    parser.add_argument('--out', required=False)

    args = parser.parse_args()

    with open(args.anns) as f:
        unfiltered_anns = {k.replace(" ", ""): v for k,v in json.load(f).items()}
        anns = {k.replace(" ", ""): v for k, v in unfiltered_anns.items() if k.split(',')[1] != 'NA'}
        if args.filter:
            with open(args.filter) as f:
                filter = json.load(f)
            anns = {k: v for k, v in anns.items() if k not in filter}
        if args.rel_blacklist:
            blacklisted_rels = set(args.rel_blacklist.split(","))
            anns = {k: v for k, v in anns.items() if k.split(",")[1] not in blacklisted_rels}


    with open(args.preds) as f:
        preds = {k.replace(" ", ""): v for k,v in json.load(f).items()}
        preds = {k: v for k, v in preds.items() if k in anns}
        if args.filter:
            with open(args.filter) as f:
                filter = json.load(f)
            preds = {k: v for k, v in preds.items() if k not in filter}
        if args.rel_blacklist:
            blacklisted_rels = set(args.rel_blacklist.split(","))
            preds = {k: v for k, v in preds.items() if k.split(",")[1] not in blacklisted_rels}

    entities = set()
    if args.pathway:
        with open(args.pathway) as f:
            pw_triples = json.load(f)
            for triple in pw_triples:
                entities.add(triple.split(',')[0])
                entities.add(triple.split(',')[2])

        anns = {k: v for k, v in anns.items() if k.split(',')[0] in entities and k.split(',')[2] in entities}
        preds = {k: v for k, v in preds.items() if k.split(',')[0] in entities and k.split(',')[2] in entities}

    if args.augment_preds:
        preds = augment_preds(preds)

    df = defaultdict(list)

    if args.baseline:
        with open(args.baseline) as f:
            baseline  = json.load(f)
        baseline = {k: v for k, v in baseline.items() if k.split(',')[1] != 'NA'}
        # baseline = augment_preds(baseline)
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

        print(relation)
        if not rel_preds:
            print(f"No predictions for {relation}")
            print()
            continue
        res = evaluate_relations(rel_anns, rel_preds, baseline)
        pprint({k: v for k,v in res.items() if 'vals' not in k})
        print()

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


