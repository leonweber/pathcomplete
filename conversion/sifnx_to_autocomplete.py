import argparse
from collections import defaultdict
from pathlib import Path

import json
import numpy as np
from networkx.utils import UnionFind
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .sifnx_to_uniprot import get_mapping_from_lines


def jaccard_sim(set1, set2):
    return len(set1 & set2)/len(set1 | set2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    pw_to_tripleset = defaultdict(set)
    with args.input.open() as f:
        lines = f.readlines()

        uniprot_mapping = get_mapping_from_lines(lines)

        for line in lines:
            line = line.strip()
            fields = line.split('\t')
            if len(fields) < 5:
                continue
            e1, r, e2 = fields[:3]
            if e1 in uniprot_mapping and e2 in uniprot_mapping:
                e1 = uniprot_mapping[e1]
                e2 = uniprot_mapping[e2]

                pw_ids = fields[5].split(';')

                triple = f"{e1},{r},{e2}"

                for pw_id in pw_ids:
                    pw_to_tripleset[pw_id].add(triple)



    uf = UnionFind()
    n_duplicates = 0
    pathways = sorted(pw_to_tripleset)
    for pw_id1, tripleset1 in tqdm(pw_to_tripleset.items(), total=len(pw_to_tripleset)):
        for pw_id2, tripleset2 in pw_to_tripleset.items():
            if pw_id1 == pw_id2:
                continue

            _ = uf[pw_id1] # get has side effect (adding to UF)
            _ = uf[pw_id2]
            sim = jaccard_sim(tripleset1, tripleset2)
            if sim == 1:
                uf.union(pw_id1, pw_id2)

    uf = UnionFind()
    pathways = sorted(set(uf[pw] for pw in pathways))
    for pw_id1 in tqdm(pathways):
        for pw_id2 in pathways:
            if pw_id1 == pw_id2:
                continue
            tripleset1 = pw_to_tripleset[pw_id1]
            tripleset2 = pw_to_tripleset[pw_id2]

            _ = uf[pw_id1] # get has side effect (adding to UF)
            _ = uf[pw_id2]
            sim = jaccard_sim(tripleset1, tripleset2)
            if sim > args.threshold:
                uf.union(pw_id1, pw_id2)


    group_to_id = {}
    groups = []
    splitter1 = GroupShuffleSplit(train_size=0.6, n_splits=1)
    splitter2 = GroupShuffleSplit(train_size=1/4, n_splits=1)
    for pw_id in pathways:
        group = uf[pw_id]
        if group not in group_to_id:
            group_to_id[group] = len(group_to_id)
        group_id = group_to_id[group]

        groups.append(group_id)

    pathways = np.array(pathways)
    groups = np.array(groups)
    train, rest = list(splitter1.split(pathways, groups=groups))[0]

    rest_groups = groups[rest]
    rest_pws = pathways[rest]
    dev, test = list(splitter2.split(rest_pws, groups=rest_groups))[0]

    train_pws = pathways[train]
    dev_pws = rest_pws[dev]
    test_pws = rest_pws[test]

    print(f"Found {len(train)/len(pathways)}, {len(dev)/len(pathways)}, {len(test)/len(pathways)} split, n={len(pathways)}")

    with open(str(args.out) + '.train', 'w') as f:
        json.dump({pw: sorted(pw_to_tripleset[pw]) for pw in train_pws}, f, indent=1)
    with open(str(args.out) + '.dev', 'w') as f:
        json.dump({pw: sorted(pw_to_tripleset[pw]) for pw in dev_pws}, f, indent=1)
    with open(str(args.out) + '.test', 'w') as f:
        json.dump({pw: sorted(pw_to_tripleset[pw]) for pw in test_pws}, f, indent=1)
