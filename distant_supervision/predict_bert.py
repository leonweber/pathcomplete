import argparse
import json
from collections import deque, defaultdict
from pathlib import Path
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import DistantBertDataset
from train_bert import MODEL_TYPES
from model import aggregate_provenance_predictions

def predict(dataset, model, args):
    dataloader = DataLoader(dataset,  batch_size=1)
    data_it = tqdm(dataloader, desc="Predicting", total=len(dataset))
    y_pred, y_true = deque(maxlen=1000), deque(maxlen=1000)
    predictions = defaultdict(dict)

    for batch in data_it:
        model.eval()
        batch = {k: v.squeeze(0).to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            logits, meta = model(**batch)

        e1, e2 = batch['entity_ids']

        e1 = dataset.file['id2entity'][e1].decode()
        e2 = dataset.file['id2entity'][e2].decode()

        provenance = aggregate_provenance_predictions(meta['alphas'], batch['pmids'])

        for i, logit in enumerate(logits):
            rel = dataset.file['id2label'][i].decode()
            score = torch.sigmoid(logit).item()
            predictions[f"{e1},{rel},{e2}"]['score'] = score

            predictions[f"{e1},{rel},{e2}"]['provenance'] = {}
            for pmid, score in provenance.items():
                predictions[f"{e1},{rel},{e2}"]['provenance'][str(pmid)] = score.detach().item()

        if 'labels' in batch:
            y_pred.append(logits.cpu().detach().numpy())
            y_true.append(batch['labels'].cpu().numpy())
            ap = average_precision_score(np.vstack(y_true), np.vstack(y_pred), average='micro')
            data_it.set_postfix_str(f"ap: {ap}")

    return predictions



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model_path', required=True, type=Path)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    train_args = torch.load(args.model_path/'training_args.bin')

    dataset = DistantBertDataset(
        args.input,
        max_bag_size=train_args.max_bag_size,
        max_length=train_args.max_length,
        ignore_no_mentions=train_args.ignore_no_mentions
    )

    model = MODEL_TYPES[train_args.model_type](train_args.bert, args=train_args)
    model.load_state_dict(torch.load(args.model_path/'weights.th'))
    model.to(args.device)

    predictions = predict(dataset=dataset, model=model, args=train_args)

    with args.output.open("w") as f:
        json.dump(predictions, f)


