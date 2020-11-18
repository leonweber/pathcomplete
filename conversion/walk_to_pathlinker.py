import argparse
from pathlib import Path
from . import utils

import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--receptors', type=Path, required=True)
    parser.add_argument('--dna_binding', type=Path, required=True)
    args = parser.parse_args()

    with args.receptors.open() as f:
        receptors = set(line.strip() for line in f if line.strip())
    with args.dna_binding.open() as f:
        dna_binding = set(line.strip() for line in f if line.strip())

    with args.input.open() as f:
        data = json.load(f)

    for pw_name, geneset in data.items():
        pw_name = utils.slugify(pw_name)

        with (args.output/pw_name).open('w') as f:
            for gene in geneset:
                if gene in receptors:
                    typ = 'receptor'
                elif gene in dna_binding:
                    typ = 'tf'
                else:
                    typ = 'none'

                f.write(f"{gene}\t{typ}\n")
