import argparse
from collections import defaultdict
from pathlib import Path

import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()


    result = defaultdict(list)
    path_files = args.input.glob('*untied-ranked-edges*')

    for path_file in path_files:
        processed_nodes = set()
        name = path_file.name.split('_untied-ranked-edges')[0]
        with path_file.open() as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line = line.split("\t")
                tail, head = line[:2]
                if tail not in processed_nodes:
                    result[name].append(tail)
                if head not in processed_nodes:
                    result[name].append(head)
                processed_nodes.update((tail, head))

    with args.output.open('w') as f:
        json.dump(result, f)






