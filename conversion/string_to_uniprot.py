import argparse
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

from tqdm import tqdm


def get_ensembl2uniprot(mapping_file):
    mapping = defaultdict(set)
    for line in mapping_file:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')

        uniprot = fields[1].split('|')[0]
        ensembl = fields[2]
        mapping[ensembl].add(uniprot)

    return mapping



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--links', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--k', type=int, default=1000000)

    args = parser.parse_args()

    with args.mapping.open() as f:
        ensembl2uniprot = get_ensembl2uniprot(f)


    pairs_with_score = set()
    with args.links.open() as f_in, args.out.open('w') as f_out:
        next(f_in)
        for line in tqdm(f_in):
            fields = line.strip().split()
            if not fields:
                continue

            e1, e2, score = fields

            e1s = ensembl2uniprot[e1]
            e2s = ensembl2uniprot[e2]

            for e1 in e1s:
                for e2 in e2s:
                    pairs_with_score.add( (tuple(sorted([e1, e2])), int(score)) )

        pairs = [p[0] for p in sorted(pairs_with_score, key=itemgetter(1))][::-1][:args.k]

        for pair in pairs:
            f_out.write("\t".join(pair) + "\n")



