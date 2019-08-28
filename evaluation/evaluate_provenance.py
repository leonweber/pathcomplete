import argparse
import json
from collections import defaultdict
import pandas as pd


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


def evaluate_relations(anns, preds, entities=None):
    true_relations = set(anns)
    pred_relations = set(preds)

    if entities:
        true_relations = filter_triples(true_relations, entities)
        pred_relations = filter_triples(pred_relations, entities)

    tps = true_relations & pred_relations
    fps = pred_relations - true_relations
    fns = true_relations - pred_relations

    if len(tps) == 0:
        return 0, 0, 0, len(true_relations)

    prec = len(tps) / (len(tps) + len(fps))
    rec = len(tps) / (len(tps) + len(fns))

    if prec == 0 or rec == 0:
        return 0, 0, 0, len(true_relations)

    f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1, len(true_relations)


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
    parser.add_argument('--out', required=False)

    args = parser.parse_args()

    with open(args.anns) as f:
        anns = json.load(f)

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
    prec, rec, f1, support = evaluate_relations(anns, preds, entities)
    print(f"Relations: {prec}/{rec}/{f1}, Support: {support}")
    df['precision'].append(prec)
    df['recall'].append(rec)
    df['relation'].append('total')
    relations = set(t.split(',')[1] for t in anns)
    for relation in relations:
        rel_anns = {k: v for k, v in anns.items() if k.split(',')[1] == relation}
        rel_preds = {k: v for k, v in preds.items() if k.split(',')[1] == relation}
        prec, rec, f1, support = evaluate_relations(rel_anns, rel_preds, entities)
        print(f"{relation}: {prec}/{rec}/{f1}, Support: {support}")
        df['precision'].append(prec)
        df['recall'].append(rec)
        df['relation'].append(relation)


    prec, rec, f1, support = evaluate_pairs(anns, preds, entities)
    print(f"Pairs: {prec}/{rec}/{f1}, Support: {support}")
    df['precision'].append(prec)
    df['recall'].append(rec)
    df['relation'].append('pair')

    # prec, rec, f1, support = evaluate_provenance(anns, preds)
    # print(f"Provenance: {prec}/{rec}/{f1}, Support: {support}")

    df['data'] = [args.anns] * len(df['precision'])
    df['predictor'] = [args.preds] * len(df['precision'])

    if args.out:
        df = pd.DataFrame(df)
        df.to_csv(args.out)

