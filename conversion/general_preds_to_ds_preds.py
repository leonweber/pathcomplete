import argparse
import json
from collections import defaultdict
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    preds_by_entities = defaultdict(list)
    with args.input.open() as f:
        data = json.load(f)
        for k, v in data.items():
            e1, r, e2 = k.split(",")
            score = v["score"]
            preds_by_entities[",".join([e1, e2])].append([r, score])

    with args.output.open("w") as f:
        for entities, preds in preds_by_entities.items():
            f.write(json.dumps(
                {
                    "entities": entities.split(","),
                    "labels": preds
                }
            ) + "\n")
