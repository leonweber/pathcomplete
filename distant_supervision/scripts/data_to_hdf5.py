import argparse
import json
from math import ceil

from tqdm import tqdm
from transformers import BertTokenizer
import h5py
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def example_to_features(example, tokenizer: BertTokenizer, max_seq_len=128):
    all_token_ids = []
    all_entity_positions = []
    all_attention_masks = []

    for mention in example['mentions']:
        e1_start = None
        e1_end = None
        e2_start = None
        e2_end = None
        token_ids = []

        mention = mention[0].lower()
        for token in mention.split():
            is_e1_start = '<e1>' in token
            is_e2_start = '<e2>' in token
            is_e1_end = '</e1>' in token
            is_e2_end = '</e2>' in token
            token = token.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')

            if is_e1_start:
                e1_start = len(token_ids)
            if is_e2_start:
                e2_start = len(token_ids)

            token_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token)))

            if is_e1_end:
                e1_end = len(token_ids)
            if is_e2_end:
                e2_end = len(token_ids)

        truncate = len(token_ids) + 2 - max_seq_len
        if truncate > 0:
            truncate1 = truncate//2
            truncate2 = ceil(truncate/2)

            if truncate1 > min(e1_start, e2_start):
                continue # truncation gobbles entity start
            if truncate2 <= max(e1_end, e2_end) - len(token_ids):
                continue # truncation gobbles entity end

            token_ids = token_ids[truncate1:-truncate2]
            e1_start -= truncate1
            e2_start -= truncate1
            e1_end -= truncate2
            e2_end -= truncate2

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
        all_attention_masks.append(attention_mask)
        all_entity_positions.append([[e1_start, e1_end], [e2_start, e2_end]])

    if len(all_token_ids) > 0:
        all_token_ids = np.vstack(all_token_ids)
        all_entity_positions = np.vstack(all_entity_positions)
        all_attention_masks = np.vstack(all_attention_masks)

        return all_token_ids, all_attention_masks, all_entity_positions
    else:
        return None, None, None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--tokenizer', required=True)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    entity_dict = {}

    with open(args.input) as f:
        data = json.load(f)

    label_binarizer = MultiLabelBinarizer()
    labels = []
    entity_ids = []
    with h5py.File(args.output, "w") as f:
        for pair, example in tqdm(data.items()):
            token_ids, attention_masks, entity_positions = example_to_features(example, tokenizer)

            if token_ids is not None:
                f.create_dataset(f"token_ids/{pair}", data=token_ids, dtype='i')
                f.create_dataset(f"attention_masks/{pair}", data=attention_masks, dtype='i')
                f.create_dataset(f"entity_positions/{pair}", data=entity_positions, dtype='i')

            e1, e2 = pair.split(',')
            if e1 not in entity_dict:
                entity_dict[e1] = len(entity_dict)
            if e2 not in entity_dict:
                entity_dict[e2] = len(entity_dict)

            entity_ids.append(np.array([entity_dict[e1], entity_dict[e2]], dtype='i'))
            labels.append([l for l in example['relations'] if l != 'NA'])


        labels = label_binarizer.fit_transform(labels)
        entity_ids = np.vstack(entity_ids)

        dset = f.create_dataset(f"entity_ids", data=entity_ids, dtype='i')

        id2entity = [None for i in entity_dict]
        for entity, id_ in entity_dict.items():
            id2entity[id_] = np.string_(entity)
        id2entity = np.array(id2entity)
        dset = f.create_dataset(f"id2entity", data=id2entity)

        dset = f.create_dataset(f"labels", data=labels, dtype='i')

        id2label = np.array([np.string_(s) for s in label_binarizer.classes_.tolist()])
        dset = f.create_dataset(f"id2label", data=id2label)

