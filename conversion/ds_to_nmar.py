import argparse
import json
import os

from pathlib import Path


def transform_sentence(sentence):
    e1_start = sentence.find('<e1>')
    sentence = sentence.replace('<e1>', '')

    e1_end = sentence.find('</e1>')
    sentence = sentence.replace('</e1>', '')

    e1 = sentence[e1_start:e1_end].replace(' ', '_')
    sentence =  sentence[:e1_start] + e1 + sentence[e1_end:]

    e2_start = sentence.find('<e2>')
    sentence = sentence.replace('<e2>', '')

    e2_end = sentence.find('</e2>')
    sentence = sentence.replace('</e2>', '')

    e2 = sentence[e2_start:e2_end].replace(' ', '_')

    sentence = sentence[:e2_start] + e2 + sentence[e2_end:]

    return sentence, e1, e2


def transform_pair(pair, data):
    mentions = data['mentions']
    relations = data['relations'] if data['relations'] else ['NA']
    lines = []

    diff = len(mentions) - len(relations)
    if diff > 0:
        relations = relations + [relations[0]] * abs(diff)
    elif diff < 0:
        mentions = mentions + [mentions[0]] * abs(diff)

    for mention, relation in zip(mentions, relations):
        sentence, e1, e2 = transform_sentence(mention[0])

        lines.append("\t".join(pair + [e1, e2, relation, sentence]))

    return set(lines)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    with args.input.open() as f:
        data = json.load(f)

    os.makedirs(args.output.parent, exist_ok=True)

    with args.output.open('w') as f:
        for k, v in data.items():
            for line in transform_pair(k.split(','), v):
                f.write(line + "\n")




