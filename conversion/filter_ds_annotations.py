import argparse
import json

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--pair_blacklist', required=False, type=Path, nargs='+', default=[])
    parser.add_argument('--triple_blacklist', required=False, type=Path, nargs='+', default=[])

    args = parser.parse_args()

    pair_blacklist = set()
    for blacklist_file in args.pair_blacklist:
        with blacklist_file.open() as f:
            pair_blacklist.update(json.load(f))

    triple_blacklist = set()
    for blacklist_file in args.triple_blacklist:
        with blacklist_file.open() as f:
            triple_blacklist.update(json.load(f))

    with args.input.open() as f, args.output.open('w') as f_out:
        anns = json.load(f)

        new_anns = {}

        for k, v in anns.items():
            e1, e2 = k.split(",")
            if k in pair_blacklist:
                continue

            new_relations = []
            for rel in v['relations']:
                if f"{e1},{rel},{e2}" not in triple_blacklist:
                    new_relations.append(rel)

            v['relations'] = new_relations
            new_anns[k] = v

        json.dump(new_anns, f_out)
