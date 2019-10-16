import json
from collections import defaultdict
from typing import Dict, Tuple, Set, List
import numpy as np
from mygene import MyGeneInfo
from tqdm import tqdm

import pandas as pd
from make_dataset import geneid_to_uniprot

from utils import load_homologene, TAX_IDS


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

# homologs = load_homologene(TAX_IDS.values())


def convert_to_sifnx(relations: pd.DataFrame, event_id_to_article: Dict[str, List], mg: MyGeneInfo):
    result = {}
    for idx, relation in tqdm(relations.iterrows(), total=len(relations)):
        if relation['refined_type'] not in TYPE_MAPPING:
            continue
        if relation['negation'] == 1:
            continue

        sifnx_type = TYPE_MAPPING[relation['refined_type']]
        heads = [str(relation['source_entrezgene_id'])]
        # heads = homologs.get(head, [head])

        tails = [str(relation['target_entrezgene_id'])]
        # tails = homologs.get(tail, [tail])

        head_proteins = set()
        tail_proteins = set()
        for head in heads:
            for tail in tails:
                head = geneid_to_uniprot(head, mg)
                tail = geneid_to_uniprot(tail, mg)

                if head:
                    head_proteins.update(head)
                if tail:
                    tail_proteins.update(tail)

        for head in head_proteins:
            for tail in tail_proteins:
                confidence = relation['confidence']
                sifnx_triple = ','.join((head, sifnx_type, tail))
                for pmid in event_id_to_article[str(relation['general_event_id'])]:
                    if sifnx_triple not in result:
                        result[sifnx_triple] = {'provenance': {}}
                    meta = result[sifnx_triple]

                    meta['provenance'][pmid] = max(confidence, meta['provenance'].get(pmid, -np.inf))
                    meta['score'] = max(confidence, meta.get('score', -np.inf))


    return result


if __name__ == '__main__':
    mg = MyGeneInfo()
    mg.set_caching('mg_cache')
    relations = pd.read_csv('../data/EVEX_relations_9606.tab', sep='\t')
    event_id_to_article = defaultdict(list)
    with open('../data/EVEX_articles_9606.tab') as f:
        next(f)
        for line in f:
            event_id, pmid = line.strip().split('\t')
            pmid = pmid.split(':')[1].strip()
            event_id_to_article[event_id].append(pmid)

    preds = convert_to_sifnx(relations, event_id_to_article, mg=mg)
    with open('../data/EVEX_preds.json', 'w') as f:
        json.dump(preds, f)
