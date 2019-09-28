import json
import pandas as pd
import argparse
from pathlib import Path
import os
from tqdm import tqdm
from collections import defaultdict

def get_triples_by_pathway(df):
    pathway_to_triples = defaultdict(set)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pathways = row['PATHWAY_NAMES']
        if not isinstance(pathways, str):
            continue # ignore NANs
        pathways = pathways.split(';')
        for pathway in pathways:
            triple = (row['PARTICIPANT_A'], 
                      row['INTERACTION_TYPE'],
                      row['PARTICIPANT_B'])

            pathway_to_triples[pathway].add(triple)

    return pathway_to_triples

def triples_to_json(triples):
    nodes = set()
    links = []
    for h, r, t in triples:
        nodes.update([h,t])
        links.append({'source': h, 'target': t, 'label': r})
    nodes = [{"id": n} for n in nodes]

    return {"nodes": nodes, "links": links}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(args.input, sep='\t')
    pathway_to_triples = get_triples_by_pathway(df)

    for pathway, triples in pathway_to_triples.items():
        pathway = pathway.replace(' ', '_').replace('/', '_')
        json_struct = triples_to_json(triples)

        with open(str(args.output/pathway) + '.json', 'w') as f:
            json.dump(json_struct, f)

