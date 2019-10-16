import argparse
import json

from pathlib import Path

def evaluate(anns, preds):
    tps = 0
    fps = 0
    fns = 0
    for pw_name in anns:
        true_nodes = set(anns[pw_name]['allowed'])
        pred_nodes = set(preds.get(pw_name, []))

        tps += len(true_nodes & pred_nodes)
        fps += len(pred_nodes - true_nodes)
        fns += len(true_nodes - pred_nodes)

    p = tps / (tps + fps)
    r = tps / (tps + fns)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anns', required=True, type=Path)
    parser.add_argument('--preds', required=True, type=Path)

    args = parser.parse_args()

    with args.anns.open() as f:
        anns = json.load(f)
    with args.preds.open() as f:
        preds = json.load(f)

    print(evaluate(anns=anns, preds=preds))
