import argparse
from copy import deepcopy
import io

import mygene
import pandas as pd
from pathlib import Path

from make_dataset import hgnc_to_uniprot, get_mapping_from_df
from mygene import MyGeneInfo

def sifnx_to_uniprot(sifnx_lines):
    mg = mygene.MyGeneInfo()
    mg.set_caching('mg_cache')
    mapping = get_mapping_from_df(pd.read_csv(io.StringIO('\n'.join(sifnx_lines)), sep='\t'))

    new_lines = set()

    for line in sifnx_lines[1:]:

        if not line.strip():
            continue
        fields = line.split('\t')
        e1s = hgnc_to_uniprot(fields[0], mapping=mapping, mg=mg)
        e2s = hgnc_to_uniprot(fields[2], mapping=mapping, mg=mg)

        for e1 in e1s:
            for e2 in e2s:
                new_fields = deepcopy(fields)
                new_fields[0] = e1
                new_fields[2] = e2
                new_lines.add('\t'.join(new_fields))

    return [sifnx_lines[0]] + list(new_lines)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    with args.input.open() as f:
        sifnx_lines = f.readlines()
    new_sifnx_lines = sifnx_to_uniprot(sifnx_lines)

    with args.output.open('w') as f:
        f.writelines(new_sifnx_lines)



