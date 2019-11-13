import argparse
import json
from collections import deque, defaultdict
from pathlib import Path
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

from .dataset import DistantBertDataset
from .model import BertForDistantSupervision


def predict(dataset, model, args, data):
    dataloader = DataLoader(dataset,  batch_size=1)
    data_it = tqdm(dataloader, desc="Predicting", total=len(dataset))
    y_pred, y_true = deque(maxlen=1000), deque(maxlen=1000)

    for batch in data_it:
        model.eval()
        batch = {k: v.squeeze(0).to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            logits, meta = model(**batch)

        e1, e2 = batch['entity_ids']


        e1 = dataset.file['id2entity'][e1].decode()
        e2 = dataset.file['id2entity'][e2].decode()

        assert f"{e1},{e2}" in dataset.pairs

        prediction = {}
        prediction['entities'] = [e1, e2]

        prediction['labels'] = []
        prediction['mentions'] = data[f"{e1},{e2}"]['mentions']
        prediction['alphas'] = torch.softmax(meta['alphas'], axis=1)[:, 0].tolist()
        for i, logit in enumerate(logits):
            rel = dataset.file['id2label'][i].decode()
            score = torch.sigmoid(logit).item()
            prediction['labels'].append([rel, score])

        if 'labels' in batch:
            y_pred.append(logits.cpu().detach().numpy())
            y_true.append(batch['labels'].cpu().numpy())
            ap = average_precision_score(np.vstack(y_true), np.vstack(y_pred), average='micro')
            data_it.set_postfix_str(f"ap: {ap}")

        yield prediction




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model_path', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()

    dataset = DistantBertDataset(
        args.input,
        # max_bag_size=train_args.max_bag_size,
        # max_length=train_args.max_length,
        # ignore_no_mentions=train_args.ignore_no_mentions
        max_bag_size=100,
        max_length=None,
        ignore_no_mentions=True
    )

    model = BertForDistantSupervision.from_pretrained(args.model_path)
    model.parallel_bert = nn.DataParallel(model.bert)
    model.to(args.device)

    with args.data.open() as f:
        data = json.load(f)

    predictions = predict(dataset=dataset, model=model, args=None, data=data)

    with args.output.open("w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


