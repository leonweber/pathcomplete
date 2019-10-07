import argparse
import json

from pathlib import Path



def convert(preds, out):
    triples = {}
    for pred in preds:
        pred = json.loads(pred)
        e1, e2 = pred["entities"]
        for label, score in pred["labels"]:
            triples[f"{e1},{label},{e2}"] = {"provenance": pred["provenance"], "score": score}

    json.dump(triples, out)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)

    args = parser.parse_args()

    with args.preds.open() as preds, args.out.open('w') as out:
        convert(preds, out)