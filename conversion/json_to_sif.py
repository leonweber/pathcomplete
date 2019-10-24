import argparse
import json

from pathlib import Path


def json_to_sif(data):
    for triple in data:
        yield triple.replace(",", "\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    with args.output.open("w") as f:
        f.write("\n".join(json_to_sif(data)))
