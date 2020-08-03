import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import trange

from .load_data import SuperPathDataset

def jaccard_sim(set1, set2):
    return len(set1 & set2)/len(set1 | set2)


def get_predictions(train_data, test_nodes):
    max_pw = None
    max_sim = 0
    for k_train, v_train in train_data.pathways.items():
        sim = jaccard_sim(test_nodes, set(v_train))
        if sim > max_sim:
            max_sim = sim
            max_pw = k_train
    if max_pw:
        return train_data.pathways[max_pw]
    else:
        return None



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--interactome', type=Path, required=True)

    args = parser.parse_args()

    train_data = SuperPathDataset(args.interactome, args.train)
    test_data = SuperPathDataset(args.interactome, args.test)

    maps = []
    maps_sample = []
    for i in trange(20):
        aps = []
        aps_sample = []
        for i, k_test in enumerate(test_data.pathways):
            v_test = test_data.pathways[k_test]
            test_nodes = set(v_test)
            test_sample = test_data[i]
            indicated_test_nodes = set(np.array(test_data.nodes)[test_sample['pw_indicators'].bool().numpy()])
            non_indicated_test_nodes = set(n for n in test_nodes if n not in indicated_test_nodes)

            predictions = get_predictions(train_data, test_nodes)
            predictions_sample = get_predictions(train_data, indicated_test_nodes)

            y_true = [int(n in non_indicated_test_nodes) for n in train_data.nodes if n not in indicated_test_nodes]

            if predictions:
                y_pred = [int(n in predictions) for n in train_data.nodes if n not in indicated_test_nodes]
                aps.append(average_precision_score(y_true, y_pred))
            else:
                aps.append(0)

            if predictions_sample:
                y_pred = [int(n in predictions_sample) for n in train_data.nodes if n not in indicated_test_nodes]
                aps_sample.append(average_precision_score(y_true, y_pred))
            else:
                aps_sample.append(0)
        maps.append(np.mean(np.nan_to_num(aps)))
        maps_sample.append(np.mean(np.nan_to_num(aps_sample)))
        print()
        print(f"mAP: {np.mean(maps)} +/- {np.std(maps)}")
        print(f"mAP (sample): {np.mean(maps_sample)} +/- {np.std(maps_sample)}")



















