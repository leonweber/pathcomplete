import argparse
import json
from collections import defaultdict
from pathlib import Path

import spacy
import scispacy
from tqdm import tqdm


def preprocess_mention(mention, nlp):
    preprocessed_mention = []
    mention = mention.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "").replace('\t', ' ').replace('\n', ' ')
    doc = nlp(mention)
    for tok in doc:
        if tok.ent_iob_ == 'B' and 'PROTEIN' in tok.ent_type_:
            preprocessed_mention.append('MYPROTEINTOKEN')
            continue
        elif tok.ent_iob_ == 'I' and 'PROTEIN' in tok.ent_type_:
            continue
        else: # no protein entity
            preprocessed_mention.append(str(tok))

    return " ".join(preprocessed_mention)

def transform(data, nlp):
    pmid_count = defaultdict(int)
    coco_lines = []
    for k, v in tqdm(list(data.items())):
        e1, e2 = k.split(',')
        for mention, _, pmid in v['mentions']:
            pmid_count[pmid] += 1

            mention = preprocess_mention(mention, nlp)
            coco_line = [pmid, str(pmid_count[pmid]), str(pmid_count[pmid]),
                         e1, e2, mention, " ".join(v['relations'])]
            coco_lines.append('\t'.join(coco_line))

    return coco_lines




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    nlp = spacy.load("en_ner_jnlpba_md", disable=["parser"])
    coco_lines = transform(data, nlp)
    fasttext_lines = []
    for line in coco_lines:
        fields = line.split('\t')
        labels = [f'__label__{l}' for l in fields[6].split()]
        fasttext_line = f"{' '.join(labels)} {fields[5]}"
        fasttext_lines.append(fasttext_line)

    with open(str(args.output) + '.coco', 'w') as f:
        for line in coco_lines:
            f.write(line + "\n")

    with open(str(args.output) + '.ft', 'w') as f:
        for line in fasttext_lines:
            f.write(line + "\n")

