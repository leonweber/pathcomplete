import argparse
import itertools
import json

import numpy as np


def convert(lines):
    relations = {}
    pairs = set()
    proteins = set()
    for line in lines:
        head, relation, tail = line.strip().split('\t')

        head_members = []
        for head_member in head.split(':'):
            if head_member:
                head_members.append(head_member)

        tail_members = []
        for tail_member in tail.split(':'):
            if tail_member:
                tail_members.append(tail_member)

        for a, b in itertools.combinations(head_members, 2):
            relations[f"{a},in-complex-with,{b}"] = []
        for a, b in itertools.combinations(tail_members, 2):
            relations[f"{a},in-complex-with,{b}"] = []

        for a in head_members:
            for b in tail_members:
                if a == b:
                    continue

                proteins.add(a)
                proteins.add(b)

                pairs.add(tuple(sorted([a, b])))

                if relation in {"phosphorylation", "dephosphorylation", "ubiquitinylation"}:
                    relations[f"{a},controls-state-change-of,{b}"] = []

                if relation in {"phosphorylation", "dephosphorylation"}:
                    relations[f"{a},controls-phosphorylation-of,{b}"] = []
                elif relation == "transport":
                    relations[f"{a},controls-transport-of,{b}"] = []
                elif relation == "transcription":
                    relations[f"{a},controls-expression-of,{b}"] = []
                elif relation == "binding":
                    relations[f"{a},in-complex-with,{b}"] = []
                    relations[f"{b},in-complex-with,{a}"] = []

    for a, b in itertools.combinations(proteins, 2):
        pair = tuple(sorted([a, b]))
        if pair not in pairs:
            relations[f"{a},NA,{b}"] = []

    return relations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines(keepends=False)

    relations = convert(lines)

    with open(args.input + '.json', 'w') as f:
        json.dump(relations, f)

