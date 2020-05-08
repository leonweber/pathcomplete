import argparse
import os
from pathlib import Path

import ignite
import optuna
import torch
import wandb
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from torch import nn
from torch.utils.data import DataLoader

from . import consts
from .dataset import PC13Dataset
from .evaluation import BioNLPMetric, Evaluator, output_transform_event_node
from .model import EventExtractor

DEBUG = True

P_event_node = ignite.metrics.Precision(output_transform=output_transform_event_node, average=True,)
R_event_node = ignite.metrics.Recall(output_transform=output_transform_event_node, average=True)
F1_event_node = ignite.metrics.Fbeta(beta=1.0, output_transform=output_transform_event_node, average=True)


def _prepare_batch(batch, device=None, non_blocking=False):
    for example in batch:
        for k, v in example.items():
            if torch.is_tensor(v):
                example[k] = convert_tensor(v, device=device, non_blocking=non_blocking)

    return batch

def create_trainer(model, optimizer, device, non_blocking=False):
    loss_fn = nn.BCEWithLogitsLoss()
    if device:
        model.to(device)
    total_loss = 0

    def _update(engine, batch):
        model.train()
        batch = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        nonlocal total_loss

        assert len(batch) == 1
        loss = model.forward_loss(batch[0])
        total_loss += loss
        if engine.state.iteration % 16 == 0:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0
        return loss.item()

    engine = Engine(_update)

    return engine



def create_evaluator(model, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        assert len(batch) == 1
        batch = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        batch = batch[0]
        with torch.no_grad():
            prediction = model.predict(batch), batch
            if DEBUG:
                preds = []
                true = []
                for pred in prediction[0]['aux']['node_logits'].argmax(dim=1):
                    preds.append(model.id_to_node_type[pred.item()])
                print(preds)
                for row in prediction[1]['node_targets'] > 0:
                    row_true = []
                    for i, v in enumerate(row):
                        if v.item():
                            row_true.append(model.id_to_node_type[i])
                    true.append(row_true)
                print(row_true)


            return prediction

    engine = Engine(_inference)
    BioNLPMetric(Evaluator(eval_script=consts.PC13_EVAL_SCRIPT, data_dir=args.dev,
                           result_re=consts.PC13_RESULT_RE, verbose=True),
                 key='f1').attach(engine, name='f1')
    P_event_node.attach(engine, name='node_precision')
    R_event_node.attach(engine, name='node_recall')
    F1_event_node.attach(engine, name='node_f1')

    return engine


def objective(trial: optuna.Trial):
    # load graph data
    if not args.disable_wandb:
        wandb.init(project="events", reinit=True)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=lambda x: x)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda x: x)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # create model
    model = EventExtractor(trial, hidden_dim=100, bert=args.bert,
                           node_type_dictionary=train_dataset.node_type_to_id,
                           edge_type_dictionary=train_dataset.edge_type_to_id)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-5,
                                 weight_decay=trial.suggest_uniform('weight_decay', 0,
                                                                    0.0))

    trainer = create_trainer(model=model, optimizer=optimizer, device=device)
    ProgressBar().attach(trainer)
    evaluator = create_evaluator(model=model, device=device)
    ProgressBar().attach(evaluator)

    pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, 'f1',
                                                                     trainer)
    pruning_evaluator = create_evaluator(model=model, device=device)
    pruning_evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    def score_function(engine):
        return engine.state.metrics['f1']

    # early_stopping = ignite.handlers.EarlyStopping(patience=10,
    #                                                score_function=score_function,
    #                                                trainer=trainer)
    # pruning_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        if (engine.state.epoch+1) % 1 == 0:
            pruning_evaluator.run(dev_loader)
            if not args.disable_wandb:
                wandb.log(
                    {'dev_' + k: v for k, v in pruning_evaluator.state.metrics.items()})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        if not args.disable_wandb:
            wandb.log({'loss': trainer.state.output})

    trainer.run(train_loader, max_epochs=1000)
    evaluator.run(dev_loader)

    return evaluator.state.metrics['f1']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--bert', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('runs/test'))
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--n_trials', type=int, default=1)
    parser.add_argument('--disable_wandb', action='store_true')

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset = PC13Dataset(args.train, args.bert)
    dev_dataset = PC13Dataset(args.dev, args.bert)
    pruner = optuna.pruners.NopPruner()
    sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(direction='maximize', pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)
    trial = study.best_trial
    print()
    print()
    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # shutil.rmtree(MODEL_DIR)
