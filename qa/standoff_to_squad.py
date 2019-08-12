from pathlib import Path
import argparse
from collections import namedtuple, defaultdict
import itertools
import json

Event = namedtuple('Event', 'type start end text')

def get_entities(a1, types=None):
    entities = {}
    for line in a1:
        fields = line.split('\t')
        entity_type = fields[1].split()[0]
        if types and entity_type not in types:
            continue

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
            event_id = args[0].split(':')[1]
            for arg in args[1:]:
                if arg.startswith('Theme:'):
                    theme = arg.split(':')[1]
                    if theme in entities:
                        themes[event_id].add(theme)
    
    return events, themes
    

def gen_simple_questions(a1, a2):
    entities = get_entities(a1, types={"Gene_or_gene_product"})
    events, themes = get_simple_events(a2, entities)

    name_to_entity_id = defaultdict(set)

    for entity_id, name in entities.items():
        name_to_entity_id[name].add(entity_id)

    all_themes = set()

    question_to_answers = defaultdict(set)
    questions = []

    for event_id, event in events.items():
        for theme in themes[event_id]:
            all_themes.add(theme)
            question = f"event {entities[theme]}"
            question_to_answers[question].add((event.start, event.text))

    for entity_theme in entities:
        if entity_theme not in all_themes:
            questions.append( {
                "question": f"event {entities[entity_theme]}",
                "id": str(hash(("".join(a1), "".join(a2), entity_theme, event_id))),
                "answers": [],
                "is_impossible": True
            } )
    
    for question, answers in question_to_answers.items():
        answers = [{"text": text, "answer_start": answer_start} for text, answer_start in answers]
        questions.append({
            "question": question,
            "id": str(hash(("".join(a1), "".join(a2), question))),
            "answers": answers,
            "is_impossible": False
        })
        
        
    return questions


def standoff_to_squad(fname):
    with open(fname + '.txt') as f:
        txt = f.read().strip()
    with open(fname + '.a1') as f:
        a1 = [l.strip() for l in f]
    with open(fname + '.a2') as f:
        a2 = [l.strip() for l in f]
    
    simple_questions = gen_simple_questions(a1, a2)
    questions = list(itertools.chain(simple_questions))

    return {"qas": questions, "context": txt}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--out')
    args = parser.parse_args()

    data = Path(args.data)
    fnames = [str(fname.stem) for fname in data.glob('*.txt')]

    squad_examples = []

    for fname in fnames:
        squad_examples.append(standoff_to_squad(str(data/fname)))

    result = {
        "version": "v0.0.1",
        "data": [{
            "title": args.data,
            "paragraphs": squad_examples
        }]
    }
    
    with open(args.out, "w") as f:
        json.dump(result, f)

