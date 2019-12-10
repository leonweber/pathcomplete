import argparse
import json
from copy import deepcopy
from pathlib import Path
import scispacy
import spacy
from tqdm import tqdm


def overlaps(range1, range2):
    return (range2[1] > range1[0] >= range2[0]) or (range1[1] > range2[0] >= range2[0])


def mask_entities(mention, nlp):
    old_mention = deepcopy(mention)
    preprocessed_mention = []
    e1_start = mention.find("<e1>")
    e1_end = mention.find("</e1>")
    e2_start = mention.find("<e2>")
    e2_end = mention.find("</e2>")

    e1 = mention[e1_start+4:e1_end]
    e2 = mention[e2_start+4:e2_end]

    mention = mention.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")

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

    assert mention[e1_start:e1_end] == e1
    assert mention[e2_start:e2_end] == e2

    doc = nlp(mention)
    protein_vocab = {}
    masked_entities = []
    cur_entity = []
    have_inserted_e1 = False
    have_inserted_e2 = False
    cur_span = None
    for tok in doc:
        cur_span = (tok.idx, len(tok) + tok.idx)

        if overlaps(cur_span, (e1_start, e1_end)) and not have_inserted_e1:
            if e1 not in protein_vocab:
                protein_vocab[e1] = len(protein_vocab)
            preprocessed_mention.append(f'<e1><protein{protein_vocab[e1]}/></e1>')
            have_inserted_e1  = True
            masked_entities.append(e1)
            continue
        if overlaps(cur_span, (e2_start, e2_end)) and not have_inserted_e2:
            if e2 not in protein_vocab:
                protein_vocab[e2] = len(protein_vocab)
            preprocessed_mention.append(f'<e2><protein{protein_vocab[e2]}/></e2>')
            have_inserted_e2 = True
            masked_entities.append(e2)
            continue

        if tok.ent_iob_ == 'B' and 'PROTEIN' in tok.ent_type_:
            cur_entity.append(str(tok))
        elif tok.ent_iob_ == 'I' and 'PROTEIN' in tok.ent_type_:
            cur_entity.append(str(tok))
        else: # no protein entity
            if cur_entity:
                protein = " ".join(cur_entity)
                masked_entities.append(protein)
                if protein not in protein_vocab:
                    protein_vocab[protein] = len(protein_vocab)
                preprocessed_mention.append(f'<protein{protein_vocab[protein]}/>')
                cur_entity = []

            preprocessed_mention.append(str(tok))

    return " ".join(preprocessed_mention), masked_entities


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    masked_data = {}
    with args.input.open() as f:
        data = json.load(f)

    spacy.require_gpu()
    nlp = spacy.load("en_ner_jnlpba_md", disable=["parser"])

    for k, v in tqdm(data.items(), total=len(data)):
        masked_data[k] = deepcopy(v)
        masked_data[k]['mentions'] = []
        masked_data[k]['masked_entities'] = []

        for mention, is_direct, pmid  in v['mentions']:
            masked_mention, masked_entities = mask_entities(mention, nlp)
            masked_data[k]['mentions'].append([masked_mention, is_direct, pmid])
            masked_data[k]['masked_entities'].append(masked_entities)

    with args.output.open('w') as f:
        json.dump(masked_data, f)



