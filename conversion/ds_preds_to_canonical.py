import argparse
import json

from pathlib import Path



def convert(preds, out):
    triples = {}
    for pred in preds:
        pred = json.loads(pred)
        e1, e2 = pred["entities"]
        for label, score in pred["labels"]:
            triples[f"{e1},{label},{e2}"] = {"provenance": {}, "score": score}

    json.dump(triples, out)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    with args.input.open() as preds, args.output.open('w') as out:
        convert(preds, out)