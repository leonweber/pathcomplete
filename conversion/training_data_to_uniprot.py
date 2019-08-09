import json
import sys
import mygene
import numpy as np

with open(sys.argv[1]) as f_in, open(sys.argv[2], 'w') as f_out:
    data = json.load(f_in)
    mg = mygene.MyGeneInfo()
    triples = [t.split(',') for t in data]
    __import__('pdb').set_trace()
    entities = set()
    for triple in triples:
        entities.update((triple[0], triple[2]))
    mapping = {e: None for e in entities}
    scores = {e: np.float('-inf') for e in entities}


    mg_result = mg.querymany(entities, scopes='entrezgene', fields='uniprot', species='human')
    for res in mg_result:
        if '_score' not in res:
            continue

        if res['_score'] > scores[res['query']]:
            if 'uniprot' in res and 'Swiss-Prot' in res['uniprot']:
                scores[res['query']] = res['_score']
                uniprot_ids = res['uniprot']['Swiss-Prot']
                if isinstance(uniprot_ids, list):
                    uniprot_id = uniprot_ids[0]
                else:
                    uniprot_id = uniprot_ids
                mapping[res['query']] = uniprot_id
    n_failed = 0
    for e1, r, e2 in triples:
        if mapping[e1] and mapping[e2]:
            f_out.write("\t".join([mapping[e1], r, mapping[e2]]))
            f_out.write("\n")
        else:
            n_failed += 1
    print("Failed: ", n_failed)

    


