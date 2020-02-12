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

    additional_blacklist = set()
    for triple in triple_blacklist:
        triple = triple.split(",")
        if 'complex' in triple[1]:
            new_triple = f"{triple[2]},{triple[1]},{triple[0]}"
            additional_blacklist.add(new_triple)
    triple_blacklist.update(additional_blacklist)

    with args.input.open() as f, args.output.open('w') as f_out:
        for line in f:
            pred = json.loads(line)

            new_preds = []
            for label, score in pred['labels']:
                if f"{pred['entities'][0]},{label},{pred['entities'][1]}" not in triple_blacklist:
                    new_preds.append([label, score])
            pred['labels'] = new_preds

            if ','.join(pred['entities']) not in pair_blacklist:
                f_out.write(json.dumps(pred) + '\n')


