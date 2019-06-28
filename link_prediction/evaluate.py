import argparse
import json
import numpy as np
import mygene
from pprint import pprint


def augment_anns(anns):
    augmented_anns = list(anns)
    for pred in anns:
        e1, r, e2 = pred.split(',')
        if r in {'controls-phosphorylation-of', 'controls-transport-of'}:
            augmented_anns.append(f"{e1},controls-state-change-of,{e2}")
        if r == 'in-complex-with':
            augmented_anns.append(f"{e2},in-complex-with,{e1}")

    return augmented_anns


def augment_preds(preds):
    augmented_preds = list(preds)
    for e1, r, e2 in preds:
        if r in {'controls-phosphorylation-of', 'controls-transport-of'}:
            augmented_preds.append((e1,"controls-state-change-of",e2))
        if r == 'in-complex-with':
            augmented_preds.append((e2,"in-complex-with",e1))

    return augmented_preds

def get_entrez2hgnc(genes):
    mg = mygene.MyGeneInfo()
    res = mg.querymany(genes, fields=['symbol'])
    mapping = {} 
    for r in res:
        mapping[r['query']] = r['symbol']

    return mapping

def evaluate(preds, anns, pathway):
    ranks = []
    anns = set(augment_anns(anns))
    known_triples = set(tuple(ann.split(',')) for ann in anns)
    pathway = set(pathway)
    pathway_nodes = set()
    pathway_nodes.update(p.split(',')[0] for p in pathway)
    pathway_nodes.update(p.split(',')[2] for p in pathway)
    print(f"{len(pathway_nodes)} pathway nodes")

    entrez2hgnc = get_entrez2hgnc(pathway_nodes)
    
    preds = [(triple, score) for triple, score in preds if triple[0] in pathway_nodes and triple[2] in pathway_nodes]
    preds = sorted(preds, key=lambda x: x[1])[::-1]
    hits = []
    new_triples = [t for t, _ in preds if tuple(t) not in known_triples and t[0] != t[2] and t[1]]
    hgnc_preds = []
    for pred in new_triples[:50]:
        hits.append(f"{pred[0]},{pred[1]},{pred[2]}" in pathway)
        hgnc_preds.append((entrez2hgnc[pred[0]], pred[1], entrez2hgnc[pred[2]]))


    known = set()
    trues = set()
    total_ranks = []
    for pw_triple in pathway:
        pw_e1, pw_r, pw_e2 = pw_triple.split(',')
        filtered_preds = [(pred[2], score) for pred, score in preds if pred[0] == pw_e1 and pred[1] == pw_r and f"{pred[0]},{pred[1]},{pred[2]}" not in anns]
        if pw_triple in anns:
            known.add(pw_triple)
            continue

        try:
            total_ranks.append(new_triples.index([pw_e1, pw_r, pw_e2]))
        except ValueError:
            total_ranks.append(None)

        rank = None
        current_rank = 1
        for pred, score in filtered_preds:
            if pred not in pathway_nodes:
                continue
            
            if pred == pw_e2:
                trues.add(pw_triple)
                rank = current_rank
                    
            current_rank += 1
            if current_rank > 3:
                break

        if rank is not None:
            ranks.append(rank)
    print(f"Known: {len(known)}/{len(pathway)}")
    print(f"Found: {len(ranks)}/{len(pathway) - len(known)}, Recall: {(len(ranks) + len(known))/len(pathway)}")
    print(f"Mean Rank: {sum(ranks)/len(ranks)}")
    print(f"TPs: {trues}")
    print(f"FNs: {pathway - (known | trues)}")
    print(total_ranks)
    pprint(hgnc_preds)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True)
    parser.add_argument('--anns', required=True)
    parser.add_argument('--pathway', required=True)
    args = parser.parse_args()

    with open(args.preds) as f:
        preds = json.load(f)

    with open(args.anns) as f:
        anns = json.load(f)

    with open(args.pathway) as f:
        pathway = json.load(f)

    evaluate(preds, anns, pathway)
