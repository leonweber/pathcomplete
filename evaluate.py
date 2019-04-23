import argparse
import json


def evaluate_relations(anns, preds):
    true_relations = set(anns)
    pred_relations = set(preds)

    tps = len(true_relations & pred_relations)
    fps = len(pred_relations - true_relations)
    fns = len(true_relations - pred_relations)


    prec = tps / (tps + fps)
    rec = tps / (tps + fns)

    if prec == 0 or rec == 0:
        return 0, 0, 0, len(true_relations)

    f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1, len(true_relations)


def evaluate_pairs(anns, preds):
    true_pairs = set((t.split(',')[0], t.split(',')[2]) for t in anns)
    pred_pairs = set((t.split(',')[0], t.split(',')[2]) for t in preds)

    tps = len(true_pairs & pred_pairs)
    fps = len(pred_pairs - true_pairs)
    fns = len(true_pairs - pred_pairs)


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

    args = parser.parse_args()

    with open(args.anns) as f:
        anns = json.load(f)

    with open(args.preds) as f:
        preds = json.load(f)

    prec, rec, f1, support = evaluate_relations(anns, preds)
    print(f"Relations: {prec}/{rec}/{f1}, Support: {support}")

    prec, rec, f1, support = evaluate_pairs(anns, preds)
    print(f"Pairs: {prec}/{rec}/{f1}, Support: {support}")

    prec, rec, f1, support = evaluate_provenance(anns, preds)
    print(f"Provenance: {prec}/{rec}/{f1}, Support: {support}")

