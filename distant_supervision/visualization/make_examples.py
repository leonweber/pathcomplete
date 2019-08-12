import sys
import json
import tqdm

N_EXAMPLES = 100


with open(sys.argv[1]) as f:
    e1 = None
    e2 = None
    supervision_type = None
    rels = None
    mentions = None
    examples = []
    # Lines are assumed to be sorted by entity1/entity2/relation_type and `distant' has to be before `direct'
    for lino, line in enumerate(tqdm.tqdm(f.readlines())):
        line = line.strip()
        new_e1, new_e2, _, _, rel, m, new_supervision_type = line.strip().split("\t")
        assert new_supervision_type in ['direct', 'distant']
        if new_e1 != e1 or new_e2 != e2 or supervision_type == 'direct':
            # new entity pair

            if mentions and 'PAIR_NOT_FOUND' not in list(mentions)[0] and 'NA' not in rels:
                examples.append({'e1': e1, 'e2': e2, 'mentions': "\n".join(mentions), "rel": " ".join(rels)})

            e1 = new_e1
            e2 = new_e2
            rels = set([rel])
            mentions = set([m])
            supervision_type = new_supervision_type
        else:
            # same pair of entities, just add the relation and the mention
            rels.add(rel)
            mentions.add(m)

with open('demo/src/examples.json', 'w') as f:
    json.dump(examples[:N_EXAMPLES], f)
