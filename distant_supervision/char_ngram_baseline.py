import argparse
import json
import numpy as np
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from tqdm import tqdm

from pathlib import Path

def transform(data):
    all_entities = []
    all_labels = []
    for k, v in data.items():
        # entities = []
        # for mention, _, _ in v['mentions']:
        #     e1 = mention[mention.find('<e1>'):mention.find('</e1>')]
        #     e2 = mention[mention.find('<e2>'):mention.find('</e2>')]
        #
        #     entities.append(e1)
        #     entities.append(e2)
        all_entities.append(" ".join(itertools.chain(*v['masked_entities'])))
        all_labels.append([r for r in v['relations'] if r != 'NA'])

    return all_entities, all_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)

    args = parser.parse_args()


    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
    label_encoder = MultiLabelBinarizer()

    with args.train.open() as f:
        train_data = {k: v for k, v in json.load(f).items() if v['mentions']}
    train_entities, train_labels = transform(train_data)

    print("Transforming train...")
    X_train = count_vectorizer.fit_transform(train_entities)
    y_train = label_encoder.fit_transform(train_labels)

    model = RandomForestClassifier(n_estimators=100, n_jobs=10)

    print("fitting...")
    model.fit(X_train, y_train)

    with args.test.open() as f:
        test_data = {k: v for k, v in json.load(f).items() if v['mentions']}
    test_entities, test_labels = transform(test_data)
    print("Transforming test...")
    X_test = count_vectorizer.transform(test_entities)

    y_pred = np.array(model.predict_proba(X_test)).swapaxes(0, 1)[:, :, 1]


    pred_lines = []
    for pair, scores in zip(test_data, y_pred):
        result = {'labels': [[label, score] for label, score in zip(label_encoder.classes_, scores)],
                  'entities': pair.split(',')}
        pred_lines.append(json.dumps(result) + '\n')

    with open('debug_preds.txt', 'w') as f:
        f.writelines(pred_lines)


