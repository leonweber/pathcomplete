import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tokenizers import BertWordPieceTokenizer
from bert_serving.client import BertClient
import h5py
from tqdm import tqdm
from transformers import BertModel


def example_to_features(example: dict, bert_client: BertClient,
                        tokenizer: BertWordPieceTokenizer,
                        max_seq_len: int = 254) -> np.array:
    n_failed = 0
    all_tokens = []
    all_e1_starts = []
    all_e2_starts = []
    for mention_fields in example['mentions']:
        e1_start = None
        e1_end = None
        e2_start = None
        e2_end = None
        tokens = []

        mention = mention_fields[0].lower()
        for token in mention.split():
            is_e1_start = '<e1>' in token
            is_e2_start = '<e2>' in token
            is_e1_end = '</e1>' in token
            is_e2_end = '</e2>' in token
            # token = token.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')

            if is_e1_start:
                e1_start = len(tokens)
            if is_e2_start:
                e2_start = len(tokens)

            tokens.extend(tokenizer.encode(token).tokens[1:-1])

            if is_e1_end:
                e1_end = len(tokens)
            if is_e2_end:
                e2_end = len(tokens)

        if len(tokens) + 2 > max_seq_len:
            n_failed += 1
            continue

        # token_ids = tokenizer.encode('[CLS]').ids + token_ids + tokenizer.encode('[SEP]').ids
        e1_start += 1
        e1_end += 1
        e2_start += 1
        e2_end += 1
        all_e1_starts.append(e1_start)
        all_e2_starts.append(e2_start)
        all_tokens.append(tokens)

    if not all_tokens:
        return None

    embs = bert_client.encode(all_tokens, is_tokenized=True)

    try:
        e1_emb = embs[np.arange(len(embs)), np.array(all_e1_starts), ...]
    except IndexError:
        return None
    try:
        e2_emb = embs[np.arange(len(embs)), np.array(all_e2_starts), ...]
    except IndexError:
        return None

    emb = np.concatenate([e1_emb, e2_emb], axis=-1)

    return emb




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bert', type=Path, required=True)
    parser.add_argument('--max_seq_len', type=int, default=254)

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    tokenizer = BertWordPieceTokenizer(str(args.bert / 'vocab.txt'))
    tokenizer.add_special_tokens(['<e1>', '</e1>', '<e2>', '</e2>'])

    bert_client = BertClient()
    with h5py.File(args.output, 'w') as f_out:
        embeddings = f_out.create_group("embeddings")
        for k in tqdm(list(sorted(data))):
            example = data[k]
            if example['mentions']:
                embs = example_to_features(
                    example=example,
                    bert_client=bert_client,
                    tokenizer=tokenizer
                )
                if embs is not None:
                    embeddings.create_dataset(k, data=embs, compression='gzip', chunks=True)
