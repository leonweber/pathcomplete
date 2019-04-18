import json
from collections import defaultdict
from typing import Dict, Tuple, Set, List

import pandas as pd

TYPE_MAPPING = {
    'Binding': 'in-complex-with',
    'Catalysis of acetylation': 'controls-state-change-of',
    'Catalysis of glycosylation': 'controls-state-change-of',
    'Catalysis of hydroxylation': 'controls-state-change-of',
    'Catalysis of methylation': 'controls-state-change-of',
    'Catalysis of phosphorylation': 'controls-phosphorylation-of',
    'Catalysis of ubiquitination': 'controls-state-change-of',
    'Regulation of expression': 'controls-expression-of',
    'Regulation of phosphorylation': 'controls-phosphorylation-of',
    'Regulation of localization': 'controls-transport-of',
    'Regulation of transcription': 'controls-expression-of'
}

PMID = str
Confidence = float
Triple = str
Prediction = Tuple[PMID, Confidence]


def convert_to_sifnx(relations: pd.DataFrame, event_id_to_article: Dict[str, List]) -> Dict[Triple, List[Prediction]]:
    sifnx_triples = defaultdict(set)
    for idx, relation in relations.iterrows():
        if relation['refined_type'] not in TYPE_MAPPING:
            continue
        if relation['negation'] == 1:
            continue

        sifnx_type = TYPE_MAPPING[relation['refined_type']]
        head = str(relation['source_entrezgene_id'])
        tail = str(relation['target_entrezgene_id'])
        confidence = relation['confidence']
        sifnx_triple = ','.join((head, sifnx_type, tail))
        for pmid in event_id_to_article[str(relation['general_event_id'])]:
            sifnx_triples[sifnx_triple].add((pmid, confidence))

    result = {}
    for k, v in sifnx_triples.items():
        result[k] = list(v)

    return result


if __name__ == '__main__':
    relations = pd.read_csv('data/EVEX_relations_9606.tab', sep='\t')
    event_id_to_article = defaultdict(list)
    with open('data/EVEX_articles_9606.tab') as f:
        next(f)
        for line in f:
            event_id, pmid = line.strip().split('\t')
            pmid = pmid.split(':')[1].strip()
            event_id_to_article[event_id].append(pmid)

    preds = convert_to_sifnx(relations, event_id_to_article)
    with open('data/EVEX_preds.json', 'w') as f:
        json.dump(preds, f)
