import argparse
import json
from pathlib import Path
import numpy as np
from evaluation.evaluate_pathway_nodes import f1_nodes, evaluate_ranking

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)

    args = parser.parse_args()

    with args.train.open() as f:
        train = json.load(f)
    with args.test.open() as f:
        test = json.load(f)

    predictions = {}

    for k_test, v_test in test.items():
        test_nodes = set(v_test)
        max_f1 = 0
        max_pw = None
        for k_train, v_train in train.items():
            _, _, f1 = f1_nodes(test_nodes, set(v_train))
            if f1 > max_f1:
                max_f1 = f1
                max_pw = k_train

        if max_pw:
            predictions[k_test] = train[max_pw]

    print(evaluate_ranking(test, predictions))










