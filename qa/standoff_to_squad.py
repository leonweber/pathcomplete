from pathlib import Path
import argparse
from collections import namedtuple, defaultdict
import itertools
import json

Event = namedtuple('Event', 'type start end text')

SIMPLE_BOOLEAN_QUESTIONS = {
    "Binding": "Does %s bind to something?",
    "Conversion": "Is %s modified?",
    "Phosphorylation": "Is %s phosphorylated?",
    "Dephosphorylation": "Is %s dephosphorylated?",
    "Acetylation": "Is %s acetylated?",
    "Deacetylation": "Is %s deacetylated?",
    "Methylation": "Is %s methylated?",
    "Demethylation": "Is %s demethylated?",
    "Ubiquitination": "Is %s ubiquitinated?",
    "Deubiquitination": "Is %s deubiquitinated?",
    "Degradation": "Does %s degrade?",
    "Regulation": "Is %s regulated?",
    "Transcription": "Is %s transcribed?",
    "Translation": "Is %s translated?",
    "Activation": "Is %s activated?",
    "Inactivation": "Is %s inactivated?",
    "Positive_regulation": "Is %s positively regulated?",
    "Negative_regulation": "Is %s negatively regulated?",
    "Gene_expression": "Is the %s expressed?",
    "Localization": "Is %s in a specific location?",
    "Transport": "Is %s transported?",
}

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
    

def gen_simple_questions(a1, a2, aggregate_answers=False):
    entities = get_entities(a1, types={"Gene_or_gene_product"})
    events, themes = get_simple_events(a2, entities)

    question_to_answers = defaultdict(set)
    questions = []

    for event_id, event in events.items():
        for theme in themes[event_id]:
            question = f"event {entities[theme]}"
            question_to_answers[question].add((event.start, event.text))

    
    for question, answers in question_to_answers.items():
        answers = [{"text": text, "answer_start": answer_start} for answer_start, text in answers]
        if aggregate_answers:
            questions.append({
                "question": question,
                "id": str(hash(("".join(a1), "".join(a2), question, "".join(a["text"] for a in answers)))),
                "answers": answers,
                "is_impossible": False
            })
        else:
            for answer in answers:
                questions.append({
                    "question": question,
                    "id": str(hash(("".join(a1), "".join(a2), question, answer["text"]))),
                    "answers": [answer],
                    "is_impossible": False
                })

    for entity_name in entities.values():
        q =  f"event {entity_name}"
        if q not in question_to_answers:
            questions.append( {
                "question": q,
                "id": str(hash(("".join(a1), "".join(a2), entity_name, event_id))),
                "answers": [],
                "is_impossible": True
            } )
        
        
    return questions

def gen_simple_boolean_questions(a1, a2, aggregate_answers=False):
    entities = get_entities(a1, types={"Gene_or_gene_product"})
    events, themes = get_simple_events(a2, entities)

    question_to_answers = defaultdict(set)
    questions = []

    for event_id, event in events.items():
        for theme in themes[event_id]:
            question = SIMPLE_BOOLEAN_QUESTIONS[event.type] % entities[theme]
            question_to_answers[question].add((event.start, event.text))

    
    for question, answers in question_to_answers.items():
        answers = [{"text": text, "answer_start": answer_start} for answer_start, text in answers]
        if aggregate_answers:
            questions.append({
                "question": question,
                "id": str(hash(("".join(a1), "".join(a2), question, "".join(a["text"] for a in answers)))),
                "answers": answers,
                "is_impossible": False
            })
        else:
            for answer in answers:
                questions.append({
                    "question": question,
                    "id": str(hash(("".join(a1), "".join(a2), question, answer["text"]))),
                    "answers": [answer],
                    "is_impossible": False
                })

    for entity_name in entities.values():
        for question_template in SIMPLE_BOOLEAN_QUESTIONS.values():
            q = question_template % entity_name
            if q not in question_to_answers:
                questions.append( {
                    "question": q,
                    "id": str(hash(("".join(a1), "".join(a2), entity_name, event_id))),
                    "answers": [],
                    "is_impossible": True
                } )

    return questions


def standoff_to_squad(fname, aggregate_answers=False):
    with open(fname + '.txt') as f:
        txt = f.read().strip()
    with open(fname + '.a1') as f:
        a1 = [l.strip() for l in f]
    with open(fname + '.a2') as f:
        a2 = [l.strip() for l in f]
    
    simple_questions = gen_simple_boolean_questions(a1, a2, aggregate_answers=aggregate_answers)
    questions = list(itertools.chain(simple_questions))

    return {"qas": questions, "context": txt}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--out')
    parser.add_argument('--aggregate_answers', action='store_true')
    args = parser.parse_args()

    data = Path(args.data)
    fnames = [str(fname.stem) for fname in data.glob('*.txt')]

    squad_examples = []

    for fname in fnames:
        squad_examples.append(standoff_to_squad(str(data/fname), aggregate_answers=args.aggregate_answers))

    result = {
        "version": "v0.0.1",
        "data": [{
            "title": args.data,
            "paragraphs": squad_examples
        }]
    }
    
    with open(args.out, "w") as f:
        json.dump(result, f)

