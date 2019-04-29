import argparse
import itertools
import json

import mygene
import numpy as np

def to_entrez(uniprot, mg):
    query_res = mg.query(uniprot, scopes='uniprot', fields='entrezgene', species='human')
    for result in query_res['hits']:
        if result['_score'] == query_res['max_score']:
            return result['entrezgene']

def convert(lines, mg):
    relations = {}
    for line in lines:
        head, relation, tail = line.strip().split('\t')

        head_members = []
        for head_member in head.split(':'):
            head_member = to_entrez(head_member, mg)
            if head_member:
                head_members.append(head_member)

        tail_members = []
        for tail_member in tail.split(':'):
            tail_member = to_entrez(tail_member, mg)
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

                if relation in {"activation", "inhibition", "phosphorylation", "transport"}:
                    relations[f"{a},controls-state-change-of,{b}"] = []

                if relation == "phosphorylation":
                    relations[f"{a},controls-phosphorylation-of,{b}"] = []
                elif relation == "transport":
                    relations[f"{a},controls-transport-of,{b}"] = []
                elif relation == "transcription":
                    relations[f"{a},controls-expression-of,{b}"] = []

    return relations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.read().splitlines(keepends=False)

    mg = mygene.MyGeneInfo()
    relations = convert(lines[1:], mg)

    with open(args.input + '.json', 'w') as f:
        json.dump(relations, f)

