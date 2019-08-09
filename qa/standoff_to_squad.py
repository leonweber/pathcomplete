from pathlib import Path
import argparse
from collections import namedtuple

Event = namedtuple('Event', 'type start end text')

def get_entities(a1):
    entities = {}
    for line in a1:
        fields = line.split('\t')
        entities[fields[0]] = fields[-1]
    
    return entities

def get_simple_events(a2, entities):
    events = {}
    themes = defaultdict(set)
    for line in a2:
        fields = line.split('\t')
        if fields[0].startswith('T'):
            type_, start, end = fields[1].split()
            events[fields[0]] = Event(type_, int(start), int(end), fields[2])
        elif fields[0].startswith('E'):
            args = fields[1].split()
            event_id = args[0].split(':')[0]
            for arg in args[1:]:
                if arg.startswith('Theme:'):
                    theme = arg.split(':')[0]
                    if theme in entities:
                        themes[event_id].append(theme)
    
    return events, themes





def gen_simple_questions(entities, a2):
    for line in a2:


def standoff_to_squad(fname):
    with open(fname + '.txt') as f:
        txt = [l.strip for l in f]
    with open(fname + '.a1') as f:
        a1 = [l.strip for l in f]
    with open(fname + '.a2') as f:
        a2 = [l.strip for l in f]
    
    entities = get_entities(a1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--out')
    args = parser.parse_args()

    data = Path(args.data)
    fnames = [str(fname.stem) for fname in data.glob('*.txt')]

    squad_examples = []

    for fname in fnames:
        squad_examples.extend(standoff_to_squad(fname))

