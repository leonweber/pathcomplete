import argparse
import logging
import random
import numpy as np
import torch
from torch import nn
import wandb

from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
from transformers import AdamW, WarmupLinearSchedule

from dataset import DistantBertDataset
from model import DistantBert

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
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
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    loss_fun = nn.BCEWithLogitsLoss()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration"):
            model.train()
            if batch['has_mentions'].sum() > 0:
                batch = {k: v.squeeze(0).to(args.device) for k, v in batch.items()}
                logits = model(**batch)
                loss = loss_fun(logits, batch['labels'].float())

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--seed', default=5005, type=int)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
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

    args = parser.parse_args()

    wandb.init(project="distant_paths")


    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning(f"n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    train_dataset = DistantBertDataset(
        args.train,
        max_bag_size=5,
        max_length=128
    )
    dev_dataset = DistantBertDataset(
        args.dev,
        max_bag_size=5,
        max_length=128
    )

    model = DistantBert(args.model, n_classes=train_dataset.n_classes)
    model.to(args.device)

    wandb.watch(model)

    train(args, train_dataset=train_dataset, model=model)
