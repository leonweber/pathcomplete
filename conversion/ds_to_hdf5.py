import argparse
import json
from math import ceil

from pathlib import Path
from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
import h5py
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from distant_supervision.run_bionlp import BioNLPProcessor


def example_to_features(example, tokenizer: BertWordPieceTokenizer, max_seq_len=256):
    all_token_ids = []
    all_entity_positions = []
    all_attention_masks = []
    all_directs = []
    all_pmids = []
    n_failed = 0

    for mention_fields in example['mentions']:
        e1_start = None
        e1_end = None
        e2_start = None
        e2_end = None
        token_ids = []


        mention = mention_fields[0].lower()
        for token in mention.split():
            is_e1_start = '<e1>' in token
            is_e2_start = '<e2>' in token
            is_e1_end = '</e1>' in token
            is_e2_end = '</e2>' in token
            # token = token.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')

            if is_e1_start:
                e1_start = len(token_ids)
            if is_e2_start:
                e2_start = len(token_ids)

            token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)))

            if is_e1_end:
                e1_end = len(token_ids)
            if is_e2_end:
                e2_end = len(token_ids)

        if len(token_ids) + 2 > max_seq_len:
            n_failed += 1
            continue
        # while len(token_ids) + 2 > max_seq_len:
        #     left_buffer = min(e1_start, e2_start)
        #     right_buffer = min(e1_end, e2_end)
        #     if left_buffer > right_buffer:
        #         token_ids = token_ids[1:]
        #         e1_start -= 1
        #         e2_start -= 1
        #         e1_end -= 1
        #         e2_end -= 1
        #     else:
        #         token_ids = token_ids[:-1]
        #
        # if min(e1_start, e2_start) < 0 or max(e1_end, e2_end) > max_seq_len:
        #     n_failed += 1
        #     continue

        token_ids = tokenizer.convert_tokens_to_ids(['[CLS]']) + token_ids + tokenizer.convert_tokens_to_ids(['[SEP]'])
        e1_start += 1
        e1_end += 1
        e2_start += 1
        e1_end += 1

        attention_mask = [1] * len(token_ids)
        if len(token_ids) < max_seq_len:
            attention_mask.extend([0] * (max_seq_len - len(token_ids)))
            token_ids.extend(tokenizer.convert_tokens_to_ids(['[PAD]']) * (max_seq_len - len(token_ids)))

        all_token_ids.append(token_ids)
        all_directs.append(mention_fields[1] == 'direct')
        all_pmids.append(int(mention_fields[2].strip()))
        all_attention_masks.append(attention_mask)
        all_entity_positions.append([[e1_start, e1_end], [e2_start, e2_end]])

    if len(all_token_ids) > 0:
        all_token_ids = np.array(all_token_ids)
        all_entity_positions = np.array(all_entity_positions)
        all_attention_masks = np.array(all_attention_masks)
        all_directs = np.array(all_directs)

        assert all_token_ids.shape[0] == all_entity_positions.shape[0] == all_attention_masks.shape[0] == all_directs.shape[0]
        return all_token_ids, all_attention_masks, all_entity_positions, all_pmids, all_directs, n_failed

    else:
        return None, None, None, None, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--entity_dict', type=Path)
    parser.add_argument('--max_seq_len', type=int, default=256)
    args = parser.parse_args()
    tokenizer = BertWordPieceTokenizer(args.tokenizer)
    tokenizer.add_special_tokens({ 'additional_special_tokens': ['<e1>','</e1>', '<e2>', '</e2>'] +
                                                                [f'<protein{i}/>' for i in range(47)]})

    with open(args.input) as f:
        data = json.load(f)

    entity_dict = {}
    if args.entity_dict:
        with args.entity_dict.open() as f:
            for line in f:
                id, item = line.strip().split('\t')
                entity_dict[item] = int(id)
    else:
        for pair in data:
            e1, e2 = pair.split(',')
            if e1 not in entity_dict:
                entity_dict[e1] = len(entity_dict)
            if e2 not in entity_dict:
                entity_dict[e2] = len(entity_dict)

    entities = [None for _ in entity_dict]
    for item, id in entity_dict.items():
        entities[id] = item

    labels = BioNLPProcessor.get_labels()


    label_binarizer = MultiLabelBinarizer(classes=labels)
    labels = []
    entity_ids = []
    total_fails = 0
    with h5py.File(args.output, "w") as f:
        data_it = tqdm(data.items())
        for pair, example in data_it:
            token_ids, attention_masks, entity_positions, pmids, is_direct, n_failed = example_to_features(example,
                                                                                                           tokenizer,
                                                                                                           max_seq_len=args.max_seq_len)

            if token_ids is not None:
                f.create_dataset(f"token_ids/{pair}", data=token_ids, dtype='i')
                f.create_dataset(f"attention_masks/{pair}", data=attention_masks, dtype='i')

                f.create_dataset(f"entity_positions/{pair}", data=entity_positions, dtype='i')
                f.create_dataset(f"pmids/{pair}", data=pmids)
                f.create_dataset(f"is_direct/{pair}", data=is_direct, dtype='bool')
                total_fails += n_failed
                data_it.set_postfix_str(f"fails: {total_fails}")

            e1, e2 = pair.split(',')

            entity_ids.append(np.array([entity_dict[e1], entity_dict[e2]], dtype='i'))
            labels.append([l for l in example['relations'] if l != 'NA'])

        labels = label_binarizer.fit_transform(labels)
        entity_ids = np.array(entity_ids)

        dset = f.create_dataset(f"entity_ids", data=entity_ids, dtype='i')

        id2entity = [None for i in entity_dict]
        for entity, id_ in entity_dict.items():
            id2entity[id_] = np.string_(entity)
        id2entity = np.array(id2entity)
        dset = f.create_dataset(f"id2entity", data=id2entity)

        dset = f.create_dataset(f"labels", data=labels, dtype='i')

        id2label = np.array([np.string_(s) for s in label_binarizer.classes_.tolist()])
        dset = f.create_dataset(f"id2label", data=id2label)
