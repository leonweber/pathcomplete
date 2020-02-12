import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--background', type=Path, required=True, nargs="+")

    args = parser.parse_args()

    with args.annotations.open() as f:
        anns = json.load(f)

    bg = set()
    for bg_file in args.background:
        with bg_file.open() as f:
            bg.update(json.load(f))

    for ann in anns:
        e1, r, e2 = ann.split(",")
        if r == "NA":
            continue
        elif "complex" in r:
            if f"{e1},{r},{e2}" not in bg and f"{e2},{r},{e1}" not in bg:
                print(ann)
        else:
            if ann not in bg:
                print(ann)
