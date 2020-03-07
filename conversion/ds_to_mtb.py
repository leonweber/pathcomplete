import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import spacy
import scispacy
from tqdm import tqdm

from .ds_tag_entities import overlaps

POS_EXAMPLES = 1e6
NEG_EXAMPLES = 1e6


def mask_entities_stochastically(mention, nlp, mask_prob=0.7, clear_prob=0.1):
    preprocessed_mention = []
    e1_start = mention.find("<e1>")
    e1_end = mention.find("</e1>")
    e2_start = mention.find("<e2>")
    e2_end = mention.find("</e2>")

    e1 = mention[e1_start + 4:e1_end]
    e2 = mention[e2_start + 4:e2_end]

    mention = mention.replace("<e1>", "").replace("</e1>", "").replace("<e2>",
                                                                       "").replace(
        "</e2>", "")
    clear = random.uniform(0, 1) < clear_prob

    e1_end -= 4
    e2_end -= 4

    new_e1_start = e1_start
    new_e1_end = e1_end
    new_e2_start = e2_start
    new_e2_end = e2_end
    if e1_start > e2_start:
        new_e1_start = new_e1_start - 4
    if e1_start > e2_end:
        new_e1_start = new_e1_start - 5
    if e1_end > e2_start:
        new_e1_end = new_e1_end - 4
    if e1_end > e2_end:
        new_e1_end = new_e1_end - 5

    if e2_start > e1_start:
        new_e2_start = new_e2_start - 4
    if e2_start > e1_end:
        new_e2_start = new_e2_start - 5
    if e2_end > e1_start:
        new_e2_end = new_e2_end - 4
    if e2_end > e1_end:
        new_e2_end = new_e2_end - 5

    e1_start = new_e1_start
    e1_end = new_e1_end
    e2_start = new_e2_start
    e2_end = new_e2_end

    doc = nlp(mention)
    cur_entity = []
    have_inserted_e1 = False
    have_inserted_e2 = False
    for tok in doc:
        cur_span = (tok.idx, len(tok) + tok.idx)

        if overlaps(cur_span, (e1_start, e1_end)) and not have_inserted_e1:
            if not clear and random.uniform(0, 1) < mask_prob:
                preprocessed_mention.append(f'<e1>[BLANK]</e1>')
            else:
                preprocessed_mention.append(f'<e1>{e1}</e1>')
            have_inserted_e1 = True
            continue
        if overlaps(cur_span, (e2_start, e2_end)) and not have_inserted_e2:
            if not clear and random.uniform(0, 1) < mask_prob:
                preprocessed_mention.append(f'<e2>[BLANK]</e2>')
            else:
                preprocessed_mention.append(f'<e2>{e2}</e2>')
            have_inserted_e2 = True
            continue

        if tok.ent_iob_ == 'B' and 'PROTEIN' in tok.ent_type_:
            cur_entity.append(str(tok))
        elif tok.ent_iob_ == 'I' and 'PROTEIN' in tok.ent_type_:
            cur_entity.append(str(tok))
        else:  # no protein entity
            if cur_entity:
                protein = " ".join(cur_entity)
                if not clear and random.uniform(0, 1) < mask_prob:
                    preprocessed_mention.append('[BLANK]')
                else:
                    preprocessed_mention.append(protein)
                cur_entity = []

            preprocessed_mention.append(str(tok))

    return " ".join(preprocessed_mention)


def convert_triplet(data, out, nlp):
    relevant_pairs = [k for k, v in data.items() if len(v['mentions']) >= 5]
    pairs_by_e1 = defaultdict(set)
    pairs_by_e2 = defaultdict(set)
    for k in relevant_pairs:
        e1, e2 = k.split(',')
        pairs_by_e1[e1].add(k)
        pairs_by_e2[e2].add(k)
    pairs_by_e1 = {k: np.array(list(v)) for k, v in pairs_by_e1.items()}
    pairs_by_e2 = {k: np.array(list(v)) for k, v in pairs_by_e2.items()}

    for pair in tqdm(relevant_pairs, desc="Writing pairs"):
        e1, e2 = pair.split(',')
        mentions = data[pair]['mentions']
        if len(mentions) > 5:
            mentions = random.sample(mentions, 5)
        anchor_mentions = [mask_entities_stochastically(m[0], nlp) for m in mentions]

        sampled_mentions = []
        neg_e1_pair = np.random.choice(pairs_by_e1[e1], 1)[0]
        mentions = data[neg_e1_pair]['mentions']
        if len(mentions) > 5:
            mentions = random.sample(mentions, 5)
        neg_e1_mentions = [mask_entities_stochastically(m[0], nlp) for m in mentions]

        mentions = []
        neg_e2_pair = np.random.choice(pairs_by_e2[e2], 1)[0]
        mentions = data[neg_e2_pair]['mentions']
        if len(mentions) > 5:
            mentions = random.sample(mentions, 5)
        neg_e2_mentions = [mask_entities_stochastically(m[0], nlp) for m in mentions]

        for anchor, neg1, neg2 in zip(anchor_mentions, neg_e1_mentions, neg_e2_mentions):
            out.write(f"{anchor}\t{neg}\t0\n")


def convert(data, out, nlp):
    relevant_pairs = [k for k, v in data.items() if len(v['mentions']) >= 5]
    pairs_by_e1 = defaultdict(set)
    pairs_by_e2 = defaultdict(set)
    for k in relevant_pairs:
        e1, e2 = k.split(',')
        pairs_by_e1[e1].add(k)
        pairs_by_e2[e2].add(k)
    pairs_by_e1 = {k: np.array(list(v)) for k, v in pairs_by_e1.items()}
    pairs_by_e2 = {k: np.array(list(v)) for k, v in pairs_by_e2.items()}

    for pair in tqdm(relevant_pairs, desc="Writing positive pairs"):
        mentions = data[pair]['mentions']
        if len(mentions) > 5:
            mentions = random.sample(mentions, 5)
            masked_mentions = []
            for mention in mentions:
                masked_mentions.append(mask_entities_stochastically(mention[0], nlp))

            for mention1, mention2 in itertools.combinations(masked_mentions, 2):
                out.write(f"{mention1}\t{mention2}\t1\n")

    n_neg_pairs = len(relevant_pairs) * 5

    easy_relation_pairs = []
    pbar = tqdm(total=n_neg_pairs, desc="Sampling negative pairs")
    relevant_pairs1 = relevant_pairs.copy()
    relevant_pairs2 = relevant_pairs.copy()
    while len(easy_relation_pairs) < n_neg_pairs // 2:
        random.shuffle(relevant_pairs1)
        random.shuffle(relevant_pairs2)

        for pair1, pair2 in zip(relevant_pairs1, relevant_pairs2):
            if pair1 != pair2:
                easy_relation_pairs.append((pair1, pair2))
                if len(easy_relation_pairs) >= n_neg_pairs // 2:
                    break
                pbar.update(1)

    hard_relation_pairs = []
    e1s = np.array(list(pairs_by_e1))
    e2s = np.array(list(pairs_by_e2))
    while len(hard_relation_pairs) < n_neg_pairs // 2:
        e1 = np.random.choice(e1s, 1)[0]
        pair1 = np.random.choice(pairs_by_e1[e1])
        pair2 = np.random.choice(pairs_by_e1[e1])
        if pair1 != pair2:
            hard_relation_pairs.append((pair1, pair2))
            pbar.update(1)

        e2 = np.random.choice(e2s, 1)[0]
        pair1 = np.random.choice(pairs_by_e2[e2])
        pair2 = np.random.choice(pairs_by_e2[e2])
        if pair1 != pair2:
            hard_relation_pairs.append((pair1, pair2))
            pbar.update(1)

    neg_pairs = easy_relation_pairs + hard_relation_pairs
    for pair1, pair2 in tqdm(neg_pairs, desc="Writing negative pairs"):
        mention1 = random.sample(data[pair1]['mentions'], 1)[0][0]
        mention2 = random.sample(data[pair2]['mentions'], 1)[0][0]

        masked_mention1 = mask_entities_stochastically(mention1, nlp)
        masked_mention2 = mask_entities_stochastically(mention2, nlp)

        out.write(f"{masked_mention1}\t{masked_mention2}\t0\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    nlp = spacy.load("en_ner_jnlpba_md", disable=["parser"])

    with args.input.open() as f:
        data = json.load(f)

    if args.test:
        small_data = {k: data[k] for k in random.sample(list(data), len(data) // 100)}
        data = small_data

    with args.output.open('w') as f:
        convert(data, out=f, nlp=nlp)
