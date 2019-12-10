import argparse
import json

from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--blacklist', required=True, type=Path, nargs='+')

    args = parser.parse_args()

    blacklist = set()
    for blacklist_file in args.blacklist:
        with blacklist_file.open() as f:
            blacklist.update(json.load(f))

    with args.input.open() as f, args.output.open('w') as f_out:
        for line in f:
            pred = json.loads(line)
            if ','.join(pred['entities']) not in blacklist:
                f_out.write(line)


