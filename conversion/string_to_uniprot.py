import argparse
from collections import defaultdict
from pathlib import Path


def get_ensembl2uniprot(mapping_file):
    mapping = defaultdict(set)
    for line in mapping_file:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')

        uniprot = fields[1].split('|')[0]
        ensembl = fields[2]
        mapping[ensembl] = uniprot

    return mapping



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--links', type=Path)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--out', type=Path)

    args = parser.parse_args()

    with args.mapping.open() as f:
        ensembl2uniprot = get_ensembl2uniprot(f)

    with args.links.open() as f_in, args.out.open('w') as f_out:
        next(f_in)
        for line in f_in:
            fields = line.strip().split()
            if not fields:
                continue

            e1, e2, score = fields
            if int(score) <= 700:
                continue

            try:
                e1 = ensembl2uniprot[e1]
                e2 = ensembl2uniprot[e2]

                f_out.write(f"{e1}\t{e2}\n")
            except KeyError:
                continue




