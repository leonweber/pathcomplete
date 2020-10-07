import argparse
import os
from collections import defaultdict
from pathlib import Path
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    synonym_to_cuids = defaultdict(set)
    cuid_to_synonyms = defaultdict(set)

    # os.makedirs(args.output_dir/"processed_train", exist_ok=True)

    current_ids = set()
    current_synonyms = set()
    current_namespace = None
    with open(args.input) as f:
        for lino, line in enumerate(f):
            line = line.strip()
            # if line.startswith("id") or line.startswith("alt_id"):
                # current_ids.add(line.split()[1])
            if line.startswith("name:"):
                current_ids.add(line[len("name:"):].strip())
                current_synonyms.add(line[len("name:"):].strip())
            elif line.startswith("synonym:"):
                current_synonyms.add(re.search(r'"(.+)"', line).group(1))
            elif line.startswith("namespace:"):
                current_namespace = line.split()[1]

            if not line:
                if current_ids and current_namespace == "biological_process":
                    for syn in current_synonyms:
                        synonym_to_cuids[syn].update(current_ids)
                    for cuid in current_ids:
                        cuid_to_synonyms[cuid].update(current_synonyms)
                current_ids = set()
                current_synonyms = set()

    with args.output.open("w") as f:
        for syn, cuids in synonym_to_cuids.items():
            f.write(f"{'|'.join(cuids)}||{syn}\n")

    # with (args.output_dir/"processed_train"/"1.concept").open("w") as f:
    #     for cuid, syns in cuid_to_synonyms.items():
    #         f.write(f"1||0|1||Foo||{list(syns)[0]}||{cuid}\n")
