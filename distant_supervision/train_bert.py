import argparse
import logging
import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig
from sklearn.metrics import average_precision_score
from torch import nn
import wandb

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from tqdm import trange, tqdm
from transformers import AdamW, WarmupLinearSchedule

from .predict_bert import predict
from .dataset import DistantBertDataset
from .model import BertForDistantSupervision

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, direct_dataset=None):
    model.train()
    if args.n_gpu > 1 and not hasattr(model.bert, 'module'):
        model.bert = nn.DataParallel(model.bert)
    model.to(args.device)

    if direct_dataset:
        train_dataset = ConcatDataset([train_dataset, direct_dataset])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    global_step = 0
    loss_fun = nn.BCEWithLogitsLoss()
    direct_loss_fun = nn.BCEWithLogitsLoss()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        tr_loss = 0.0
        logging_losses = []
        logging_direct_losses = []
        logging_distant_losses = []
        y_pred, y_true = deque(maxlen=100), deque(maxlen=100)
        direct_aps = []
        epoch_iterator = enumerate(train_dataloader)
        pbar = tqdm(total=len(train_dataloader) // args.gradient_accumulation_steps, desc="Batches")
        log_dict = None
        for step, batch in epoch_iterator:
            model.train()
            if args.n_gpu > 1 and not hasattr(model.bert, 'module'):
                model.bert = nn.DataParallel(model.bert)
            model.to(args.device)

            batch = {k: v.squeeze(0).to(args.device) for k, v in batch.items()}
            logits, meta = model(**batch)

            y_pred.append(logits.cpu().detach().numpy())
            y_true.append(batch['labels'].cpu().numpy())

            distant_loss = loss_fun(logits, batch['labels'].float())
            logging_distant_losses.append(distant_loss.item())

            if batch['has_direct'].item():
                direct_loss = direct_loss_fun(meta['alphas'], batch['is_direct'].float())
                logging_direct_losses.append(direct_loss.item())
            else:
                direct_loss = None

            is_direct = batch['is_direct'].cpu().numpy().ravel()
            if is_direct.sum() > 0:
                direct_ap = average_precision_score(batch['is_direct'].cpu().numpy(), meta['alphas'].cpu().detach().numpy().ravel(),
                                                    average='micro')
                direct_aps.append(direct_ap)

            if direct_loss:
                lambda_ = 0.5
                loss = lambda_ * direct_loss + (1 - lambda_) * distant_loss
            else:
                loss = distant_loss

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            logging_losses.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                pbar.update(1)

                ap = average_precision_score(np.vstack(y_true), np.vstack(y_pred), average='micro')
                log_dict = {
                    'loss': np.mean(logging_losses),
                    'direct_loss': np.mean(logging_direct_losses) if logging_direct_losses else None,
                    'distant_loss': np.mean(logging_distant_losses),
                    'distant_ap': ap,
                    'direct_map': np.mean(direct_aps) if direct_aps else None,
                }
                for k, v in meta.items():
                    if hasattr(v, 'detach'):
                        v = v.detach()
                    if hasattr(v, 'cpu'):
                        v = v.cpu().numpy()
                    if not args.disable_wandb:
                        if '_hist' in k:
                            v = wandb.Histogram(np_histogram=v)

                    log_dict[k] = v

                if not args.disable_wandb:
                    wandb.log(log_dict, step=global_step)
                pbar.set_postfix_str(f"loss: {log_dict['loss']}, ap: {ap}, dmAP: {log_dict['direct_map']}")
                logging_losses = []
                logging_direct_losses = []
                logging_distant_losses = []

        # Evaluation
        val_ap = None
        for _, val_ap in predict(dev_dataset, model): # predict yields prediction and current ap => exhaust iterator
            pass
        print("Validation AP: " + str(val_ap))
        if not args.disable_wandb:
            wandb.log({'val_distant_ap': val_ap}, step=global_step)


        # Saving
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_dir = args.output_dir / f'checkpoint-{global_step}'
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        model.to('cpu')
        if hasattr(model.bert, 'module'):
            model.bert = model.bert.module
        model.save_pretrained(output_dir)
        model.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--direct_data', default=None, type=Path)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--seed', default=5005, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--output_dir', type=Path, default=Path('runs/test'))
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_bag_size", default=None, type=int)
    parser.add_argument("--max_length", default=None, type=int)
    parser.add_argument("--tensor_emb_size", default=200, type=int)
    parser.add_argument("--subsample_negative", default=1.0, type=float)
    parser.add_argument('--ignore_no_mentions', action='store_true')
    parser.add_argument('--init_from', type=Path)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    if not args.disable_wandb:
        wandb.init(project="distant_paths")

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning(f"n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    train_dataset = DistantBertDataset(
        args.train,
        max_bag_size=args.max_bag_size,
        max_length=args.max_length,
        ignore_no_mentions=args.ignore_no_mentions,
        subsample_negative=args.subsample_negative,
        has_direct=False
    )
    dev_dataset = DistantBertDataset(
        args.dev,
        max_bag_size=args.max_bag_size,
        max_length=args.max_length,
        ignore_no_mentions=args.ignore_no_mentions,
        has_direct=False
    )
    if args.direct_data:
        direct_dataset = DistantBertDataset(
            args.direct_data,
            max_bag_size=args.max_bag_size,
            max_length=args.max_length,
            ignore_no_mentions=args.ignore_no_mentions,
            has_direct=True
        )
    else:
        direct_dataset = None

    config = BertConfig.from_pretrained(args.bert, num_labels=train_dataset.n_classes )

    model = BertForDistantSupervision.from_pretrained(args.bert,
                                                      config=config
                                                      )
    if not args.disable_wandb:
        wandb.watch(model)
        wandb.config.update(args)
    train(args, train_dataset=train_dataset, model=model, direct_dataset=direct_dataset)
