import argparse
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from events.dataset import PC13Dataset
from events.model import EventExtractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--bert', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('runs/test'))

    args = parser.parse_args()

    model = EventExtractor(train=args.train, dev=args.dev, bert=args.bert)
    model.load_state_dict(torch.load(args.weights)["state_dict"])
    model.cuda()
    train_dataset = PC13Dataset(args.train, args.bert, predict=True)
    for i in train_dataset:
        model.predict(i)
