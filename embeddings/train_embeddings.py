import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import pandas as pd
import torch
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from pytorch_lightning.callbacks.pt_callbacks import ModelCheckpoint
from pytorch_lightning.logging import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AdamW

import logging
from .model import SentenceMatcher
from .dataset import BioNLPMatchingDataset


def _prepare_batch(batch, device=None, non_blocking=False):
    return convert_tensor(batch, device=device, non_blocking=non_blocking)


def create_triplet_trainer(model, optimizer, device, loss_fn,
                           non_blocking=False,
                           output_transform=lambda anchor, pos, neg, loss: loss.item()):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        anchor, pos, neg = model(batch)
        loss = loss_fn(anchor, pos, neg)
        loss.backward()
        optimizer.step()
        return output_transform(anchor, pos, neg, loss)

    return Engine(_update)


def main(args):
    if not args.disable_wandb:
        wandb.init(project="test_embeddings", reinit=True)
    if torch.cuda.is_available():
        train_batch_size = 2 * torch.cuda.device_count()
    else:
        train_batch_size = 2
    train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size)
    model = nn.DataParallel(
        SentenceMatcher(dataset, args.bert))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = nn.TripletMarginLoss()
    trainer = create_triplet_trainer(model=model, optimizer=optimizer, device=device,
                                     loss_fn=loss_fn)
    pbar = ProgressBar()
    pbar.attach(trainer)

    if not args.disable_wandb:
        @trainer.on(Events.EPOCH_COMPLETED)
        def save(engine: Engine):
            save_dir = args.output_dir / f"epoch_{engine.state.epoch}"
            if not save_dir.exists():
                save_dir.mkdir()
            model.module.cpu().bert.save_pretrained(save_dir)
            model.to(device)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            wandb.log({'loss': trainer.state.output})

    trainer.run(train_loader, max_epochs=args.epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--bert', required=True)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not args.output_dir.exists():
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(args.bert)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})
    dataset = BioNLPMatchingDataset(pd.read_csv(args.data), tokenizer=tokenizer)
    main(args)
