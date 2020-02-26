import argparse
import itertools
from copy import deepcopy
import io

import mygene
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from .make_dataset import hgnc_to_uniprot

def get_mapping_from_lines(sifnx_lines):
    mapping = {}
    for line in tqdm(sifnx_lines):
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        if fields[1] == 'ProteinReference':
            for field in fields:
                if 'uniprot knowledgebase' in field:
                    mapping[fields[0]] = field.split(':')[1]

    return mapping

def sifnx_to_uniprot(sifnx_lines):
    mg = mygene.MyGeneInfo()
    mg.set_caching('mg_cache')
    mapping = get_mapping_from_lines(sifnx_lines)

    new_lines = set()

    for line in tqdm(sifnx_lines):
        line = line.strip()
        if 'CHEBI' in line or 'ProteinReference' in line or 'PARTICIPANT_A' in line:
            continue

        if not line.strip():
            continue
        fields = line.split('\t')
        e1 = hgnc_to_uniprot(fields[0], mapping=mapping, mg=mg)
        e2 = hgnc_to_uniprot(fields[2], mapping=mapping, mg=mg)

        if not (e1 and e2) :
            continue

        if not isinstance(e1, list):
            e1 = [e1]
        if not isinstance(e2, list):
            e2 = [e2]

        if isinstance(e1[0], list):
            e1 = list(itertools.chain(*e1))
        if isinstance(e2[0], list):
            e2 = list(itertools.chain(*e2))

        for a in e1:
            for b in e2:
                new_fields = deepcopy(fields)
                new_fields[0] = a
                new_fields[2] = b
                new_lines.add('\t'.join(new_fields) + '\n')

    return list(new_lines)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sif', action='store_true')

    args = parser.parse_args()

    with args.input.open() as f:
        sifnx_lines = f.readlines()
    new_sifnx_lines = sifnx_to_uniprot(sifnx_lines)

    if args.sif:
        new_sifnx_lines = ['\t'.join(l.split('\t')[:2]) + '\n' for l in new_sifnx_lines]

    with args.output.open('w') as f:
        f.writelines(new_sifnx_lines)



