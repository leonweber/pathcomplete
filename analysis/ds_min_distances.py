import argparse
import json
from pathlib import Path

import numpy as np
from collections import defaultdict
from tqdm import tqdm

LINE_ESTIMATE = 756831643

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    min_dist = defaultdict(lambda: np.inf)

    with args.input.open() as f:
        for line in tqdm(f, total=LINE_ESTIMATE):
            _, pair, distance, _ = line.strip().split('\t')
            min_dist[pair] = min(min_dist[pair], distance)

    with args.output.open('w') as f:
        json.dump(min_dist, f)

