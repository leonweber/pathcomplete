import argparse
import json
import pandas as pd
import numpy as np
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer
from tqdm import tqdm

from pathlib import Path

def get_entities(texts):
    entities = []
    labels = []
    for text in texts:
        e1 = text[text.find('<e1>'):text.find('</e1>')]
        e2 = text[text.find('<e2>'):text.find('</e2>')]

        entities.append(e1 + " " + e2)

    return entities

def get_labels(labels):
    new_labels = []
    for label in labels:
        if '|' in label:
            fields = label.split('|')
            rel = fields[-1]
            mods = fields[:-1]
        else:
            rel = label
            mods = []

        if mods:
            rel = 'No'
        new_labels.append(rel)

    return new_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)

    args = parser.parse_args()

    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
    label_encoder = LabelBinarizer()

    train_df = pd.read_csv(args.train)
    train_entities = get_entities(train_df.text)
    train_labels = get_labels(train_df.labels)

    print("Transforming train...")
    X_train = count_vectorizer.fit_transform(train_entities)
    y_train = label_encoder.fit_transform(train_labels)

    model = RandomForestClassifier(n_estimators=100, n_jobs=10, class_weight='balanced_subsample')

    print("fitting...")
    model.fit(X_train, y_train)

    test_df = pd.read_csv(args.test)
    test_entities = get_entities(test_df.text)
    test_labels = get_labels(test_df.labels)

    X_test = count_vectorizer.transform(test_entities)
    y_test = label_encoder.fit_transform(test_labels)
    y_pred = np.array(model.predict(X_test))

    baseline = np.zeros_like(y_test)
    baseline[:, 0] = 1
    print("Acc:", accuracy_score(y_true=y_test, y_pred=y_pred))
    print("Acc (baseline):", accuracy_score(y_true=y_test, y_pred=baseline))
    print("F1:", f1_score(y_true=y_test, y_pred=y_pred, average='macro', labels=np.arange(1, y_train.max() + 1)))


