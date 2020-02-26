import argparse
import itertools
import os
from collections import defaultdict
from pathlib import Path

import scispacy
import pandas as pd
import spacy
from tqdm import tqdm

from .standoff_to_ds import parse, Event, Theme, TYPE_MAPPING, ENTITY_TYPES, get_mention, fname_to_pmid, add_modifiers, \
    MODIFIER_MAPPING, get_possible_pairs


def add_example(e1, e2, rel_type, fname, doc, examples):
    mention = get_mention(e1=e1, e2=e2, doc=doc).replace('\t', ' ').replace('\n', ' ').strip()

    examples['text'].append(mention)
    examples['labels'].append(rel_type)
    examples['pmid'].append(fname_to_pmid(fname))

    return None


def transform(fname, nlp):
    with fname.with_suffix('.txt').open() as f:
        txt = f.read().strip()
    with fname.with_suffix('.a1').open() as f:
        a1 = [l.strip() for l in f]
    with fname.with_suffix('.a2').open() as f:
        a2 = [l.strip() for l in f]

    Event.reset()
    parse(a1 + a2)
    doc = nlp(txt)
    Event.resolve_all_ids()

    add_modifiers(a2)
    examples = defaultdict(list)

    event: Event
    for event_id, event in Event.registry.items():
        causes = []
        themes = [t for t in event.themes if isinstance(t, Theme)]

        if event.type not in TYPE_MAPPING:
            continue
        else:
            relation_type = TYPE_MAPPING[event.type][0]

        if event.modifiers:
            mod = '|'.join(sorted(MODIFIER_MAPPING[m] for m in event.modifiers))
            relation_type = mod + '|' +  relation_type

        if event.type == 'Binding' or event.type == 'Dissociation':
            for e1, e2 in itertools.combinations(event.themes, 2):
                if e1.type not in ENTITY_TYPES or e2.type not in ENTITY_TYPES:
                    continue
                add_example(e1=e1, e2=e2, rel_type=relation_type, fname=fname, doc=doc, examples=examples)
                add_example(e1=e2, e2=e1, rel_type=relation_type, fname=fname, doc=doc, examples=examples)
        else:
            causes += event.causes
            causes += event.get_regulators([])

            theme: Theme
            cause: Theme
            for theme in themes:
                for cause in causes:
                    if theme.type not in ENTITY_TYPES or cause.type not in ENTITY_TYPES:
                        continue
                    add_example(e1=cause, e2=theme, rel_type=relation_type, fname=fname, doc=doc, examples=examples)
                    add_example(e1=theme, e2=cause, rel_type=relation_type + '_r', fname=fname, doc=doc, examples=examples)

    for e1, e2 in get_possible_pairs([t for t in Theme.registry.values() if t.type in ENTITY_TYPES], dist=50):
        add_example(e1=e1, e2=e2, rel_type='No', fname=fname, doc=doc, examples=examples)
        add_example(e1=e2, e2=e1, rel_type='No', fname=fname, doc=doc, examples=examples)


    return examples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    os.makedirs(args.output.parent, exist_ok=True)

    fnames = list(args.input.glob('*.txt'))

    examples = defaultdict(list)
    nlp = spacy.load('en_core_sci_sm', disable=['tagger'])

    for fname in tqdm(fnames):
        for k, v in transform(fname, nlp=nlp).items():
            examples[k].extend(v)

    df = pd.DataFrame(examples)
    df.to_csv(args.output)
