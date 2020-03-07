import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AdamW, PreTrainedTokenizer

from .model import SentenceMatcher
from .dataset import BioNLPMatchingDataset, MTBDataset


def _prepare_batch(batch, device=None, non_blocking=False):
    return convert_tensor(batch, device=device, non_blocking=non_blocking)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
     Adapted from https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    batch_size, n_sentences, length = inputs.shape

    inputs = inputs.view(batch_size * n_sentences, length)
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, 0.15).to(inputs.device)
    special_tokens_mask = torch.zeros_like(labels)
    for special_id in tokenizer.all_special_ids:
        special_tokens_mask = special_tokens_mask + labels.eq(
            torch.tensor(special_id)).long()
    probability_matrix.masked_fill_(
        special_tokens_mask.bool().to(inputs.device),
        value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(
        torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(
        torch.full(labels.shape, 0.5)).bool().to(
        inputs.device) & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long).to(
        inputs.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    inputs = inputs.view((batch_size, n_sentences, length))
    labels = labels.view_as(inputs)
    return inputs, labels


def create_triplet_trainer(model: SentenceMatcher, optimizer, device, loss_fn,
                           non_blocking=False, loss_fn_mlm=None, mlm_weight=0.8,
                           output_transform=lambda anchor, pos, neg, loss: loss.item()):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        batch['input_ids'], mlm_labels = mask_tokens(batch['input_ids'], tokenizer)

        anchor, pos, neg, mlm_logits = model(batch)
        loss = loss_fn(anchor, pos, neg)
        if loss_fn_mlm:
            loss_mlm = loss_fn_mlm(mlm_logits.view(-1, mlm_logits.shape[-1]),
                                   mlm_labels.view(-1))
            loss = mlm_weight * loss_mlm + (1 - mlm_weight) * loss
        loss.backward()
        optimizer.step()
        return output_transform(anchor, pos, neg, loss)

    return Engine(_update)


def create_mtb_trainer(model: SentenceMatcher, optimizer, device,
                       non_blocking=False, loss_fn_mlm=None, mlm_weight=0.5,
                       output_transform=lambda anchor, pos, neg, loss: loss.item()):
    loss_fn_mtb = nn.BCEWithLogitsLoss()
    loss_fn_mlm = nn.CrossEntropyLoss()

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        batch, labels_mtb = _prepare_batch(batch, device=device,
                                           non_blocking=non_blocking)
        batch['input_ids'], mlm_labels = mask_tokens(batch['input_ids'], tokenizer)

        embs, logits_mlm = model(batch)
        logits_mtb = torch.einsum('bd,bd->b', embs[:, 0], embs[:, 1])
        loss_mtb = loss_fn_mtb(logits_mtb, labels_mtb)

        loss_mlm = loss_fn_mlm(logits_mlm.view(-1, logits_mlm.shape[-1]),
                               mlm_labels.view(-1))
        loss = mlm_weight * loss_mlm + (1 - mlm_weight) * loss_mtb
        loss.backward()
        optimizer.step()
        return loss

    return Engine(_update)


def main(args):
    if not args.disable_wandb:
        wandb.init(project="train_embeddings")
        wandb.config.update(args)
    if torch.cuda.is_available():
        train_batch_size = args.batch_size * torch.cuda.device_count()
    else:
        train_batch_size = args.batch_size
    train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
    e1_id = torch.tensor(tokenizer.convert_tokens_to_ids('<e1>'))
    e2_id = torch.tensor(tokenizer.convert_tokens_to_ids('<e2>'))
    model = nn.DataParallel(
        SentenceMatcher.from_pretrained(args.bert, e1_start_id=e1_id, e2_start_id=e2_id))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    trainer = create_mtb_trainer(model=model, optimizer=optimizer, device=device)

    pbar = ProgressBar()
    pbar.attach(trainer)

    if not args.disable_wandb:
        @trainer.on(Events.ITERATION_COMPLETED)
        def save(engine: Engine):
            if (engine.state.iteration + 1) % 10000 == 0:
                save_dir = args.output_dir / f"step_{engine.state.iteration}"
                if not save_dir.exists():
                    save_dir.mkdir()
                model.module.cpu().bert.save_pretrained(save_dir)
                model.to(device)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            wandb.log({'loss': trainer.state.output})

    @trainer.on(Events.EPOCH_STARTED)
    def prepare_data(engine: Engine):
        if args.resample:
            dataset.resample_indices()
        if args.hard_negatives:
            dataset.update_dist_matrix(model.module.bert)

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--bert', required=True)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--hard_negatives', action='store_true')
    parser.add_argument('--resample', action='store_true')
    parser.add_argument('--mlm', action='store_true')
    parser.add_argument('--test', action='store_true')
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
        {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>', '[BLANK]']})
    dataset = MTBDataset(args.data, tokenizer=tokenizer)
    main(args)
