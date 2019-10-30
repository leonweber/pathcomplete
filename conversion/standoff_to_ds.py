import itertools
import os
from bisect import bisect_right, bisect_left
from pathlib import Path
import argparse
import json

unmappable = set()

TYPE_MAPPING = {
    'Gene_expression': ['controls-expression-of'],
    'Translation': ['controls-expression-of'],
    'Transcription': ['controls-expression-of'],
    'Transport': ['controls-transport-of', 'controls-state-change-of'],
    'Phosphorylation': ['controls-phosphorylation-of', 'controls-state-change-of'],
    'Dephosphorylation': ['controls-phosphorylation-of', 'controls-state-change-of'],
    'Acetylation': ['controls-state-change-of'],
    'Deacetylation': ['controls-state-change-of'],
    'Ubiquitination': ['controls-state-change-of'],
    'Deubiquitination': ['controls-state-change-of'],
    'Hydroxylation': ['controls-state-change-of'],
    'Dehydroxylation': ['controls-state-change-of'],
    'Methylation': ['controls-state-change-of'],
    'Demethylation': ['controls-state-change-of'],
    'Binding': ['in-complex-with'],
    'Dissociation': ['in-complex-with']
}


def get_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """

    #token_ends = [len(tokens[0])]
    #for token in tokens[1:]:
        #new_end = token_ends[-1] + len(token) + 1
        #token_ends.append(new_end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return (new_start, new_end)

class Theme:
    registry = {}

    def __init__(self, id, start, end, mention):
        self.id = id
        self.start = start
        self.end = end
        self.mention = mention
        self.cause_of = []
        self.theme_of = []

    def __str__(self):
        return f"{self.id}: {self.mention}"

    def __repr__(self):
        return str(self)


    @staticmethod
    def from_line(line):
        fields = line.strip().split('\t')
        id = fields[0]
        type, start, end = fields[1].split()
        start = int(start)
        end = int(end)
        mention = fields[2]

        return Theme(start=start, end=end, id=id, mention=mention)

    def register(self):
        self.registry[self.id] = self


def get_theme_or_event(id):
    if id.startswith('E'):
        return Event.registry[id]
    elif id.startswith('T'):
        return Theme.registry[id]
    else:
        raise ValueError(id)



class Event:
    registry = {}

    @classmethod
    def resolve_all_ids(cls):
        for event in cls.registry.values():
            event.resolve_ids()

    def __init__(self, id, themes, causes, products, mention, type):
        self.id = id
        self.themes = themes
        self.causes = causes
        self.products = products

        self.theme_of = []
        self.cause_of = []

        self.mention = mention
        self.type = type

    def __str__(self):
        return f"{self.id}:{self.type} Themes: {self.themes} Causes: {self.causes}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_line(line):
        fields = line.strip().split('\t')
        id = fields[0]

        args = fields[1].split()
        type, mention = args[0].split(':')

        themes = set()
        causes = set()
        products = set()
        for arg in args[1:]:
            if arg.startswith('Theme'):
                themes.add(arg.split(':')[1])
            elif arg.startswith('Cause'):
                causes.add(arg.split(':')[1])
            elif arg.startswith('Participant'):
                continue
            elif arg.startswith('ToLoc'):
                continue
            elif arg.startswith('FromLoc'):
                continue
            elif arg.startswith('Product'):
                products.add(arg.split(':')[1])
            elif arg.startswith('Site'):
                continue
            elif arg.startswith('AtLoc'):
                continue
            else:
                raise ValueError(f"{arg}: {line}")

        return Event(id=id, themes=themes, causes=causes, mention=mention, type=type, products=products)

    def resolve_ids(self):
        resolved_themes = []
        for theme in self.themes:
            resolved_theme = get_theme_or_event(theme)
            resolved_themes.append(resolved_theme)
            resolved_theme.theme_of.append(self)
        self.themes = resolved_themes

        resolved_causes = []
        for cause in self.causes:
            resolved_cause = get_theme_or_event(cause)
            resolved_causes.append(resolved_cause)
            resolved_cause.cause_of.append(self)
        self.causes = resolved_causes

        resolved_products = []
        for product in self.products:
            resolved_product = get_theme_or_event(product)
            resolved_products.append(resolved_product)
        self.products = resolved_products

        self.mention = Theme.registry[self.mention]

    def register(self):
        self.registry[self.id] = self

    def get_regulators(self):
        regulators = []
        for event in self.theme_of:
            if 'Regulation' in event.type:
                for cause in event.causes:
                    if isinstance(cause, Theme):
                        regulators.append(cause)
                    else:
                        regulators += cause.get_regulators()
                regulators += event.get_regulators()

        return regulators



def parse(lines):
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('T'):
            Theme.from_line(line).register()
        elif line.startswith('E'):
            Event.from_line(line).register()


def get_mention(e1: Theme, e2: Theme, text: str):
    if e1.start < e2.start:
        left_ent = e1
        left_ent_tag = 'e1'
        right_ent = e2
        right_ent_tag = 'e2'
    else:
        left_ent = e2
        left_ent_tag = 'e2'
        right_ent = e1
        right_ent_tag = 'e1'

    new_text = text[:left_ent.start] + f"<{left_ent_tag}>" +\
               text[left_ent.start:left_ent.end] + f"</{left_ent_tag}>" +\
               text[left_ent.end:right_ent.start] + f"<{right_ent_tag}>" + \
               text[right_ent.start:right_ent.end] + f"</{right_ent_tag}>" +\
               text[right_ent.end:]

    assert new_text.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "") == text

    tokens = new_text.split()
    token_starts = [0]
    for token in tokens[:-1]:
        new_start = token_starts[-1] + len(token) + 1
        token_starts.append(new_start)

    left_span = get_span(left_ent.start, left_ent.end, token_starts)
    right_span = get_span(right_ent.start, right_ent.end, token_starts)

    left_boundary = max(left_span[0] - 25, 0)
    right_boundary = min(right_span[1] + 25, len(tokens))
    truncated_tokens = tokens[left_boundary:right_boundary]

    return " ".join(truncated_tokens)


def transform(fname, transformed_data):
    with open(fname + '.txt') as f:
        txt = f.read().strip()
    with open(fname + '.a1') as f:
        a1 = [l.strip() for l in f]
    with open(fname + '.a2') as f:
        a2 = [l.strip() for l in f]

    parse(a1 + a2)

    Event.resolve_all_ids()

    event: Event
    for event_id, event in Event.registry.items():
        causes = []
        themes = [t for t in event.themes if isinstance(t, Theme)]

        if event.type not in TYPE_MAPPING:
            unmappable.add(event.type)
            continue
        else:
            relation_types = TYPE_MAPPING[event.type]


        if event.type == 'Binding':
            for e1, e2 in itertools.combinations(event.themes, 2):
                transform_pair(e1, e2, relation_types, fname, transformed_data, txt)
        elif event.type == 'Dissociation':
            for e1, e2 in itertools.combinations(event.products, 2):
                transform_pair(e1, e2, relation_types, fname, transformed_data, txt)
        else:
            causes += event.causes
            causes += event.get_regulators()

            theme: Theme
            cause: Theme
            for theme in themes:
                for cause in causes:
                    transform_pair(cause, theme, relation_types, fname, transformed_data, txt)

        return transformed_data


def transform_pair(e1, e2, relation_types, fname, transformed_data, txt):
    pair = f"{e1.mention},{e2.mention}"
    if pair not in transformed_data:
        transformed_data[pair] = {
            'relations': set(),
            'mentions': []
        }
    transformed_data[pair]['relations'].update(relation_types)
    mention = get_mention(e1=e1, e2=e2, text=txt)
    transformed_data[pair]['mentions'].append(
        [mention, "direct", os.path.basename(fname).split("PMID-")[1]]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    data = Path(args.data)
    fnames = [str(fname.stem) for fname in data.glob('*.txt')]

    transformed_data = {}

    for fname in fnames:
        Event.registry = {}
        Theme.registry = {}
        transform(str(data/fname), transformed_data)

    for v in transformed_data.values():
        v['relations'] = list(v['relations'])


    with open(args.out, "w") as f:
        json.dump(transformed_data, f, indent=1)

    print(unmappable)
