import argparse
import json
from pathlib import Path

import h5py
from tqdm import tqdm
from .model import BERTEntityEmbedder



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bert', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    embedder = BERTEntityEmbedder(args.bert)

    with h5py.File(args.output, 'w') as f_out:
        embeddings_group = f_out.create_group("embeddings")
        for k in tqdm(list(sorted(data))):
            example = data[k]
            if example['mentions']:
                embs = embedder.embed([m[0] for m in example['mentions']])
                if embs is not None:
                    embeddings_group.create_dataset(k, data=embs, compression='gzip', chunks=True)


