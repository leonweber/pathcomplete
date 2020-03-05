import argparse
import shutil
from pathlib import Path

import torch
import ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Precision, Recall
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AdamW, BertModel
import optuna
import wandb

from .dataset import BioNLPClassificationDataset, BERTEntityEmbedder, \
    BERTTokenizerEmbedder, MyElmoEmbedder

DIR = Path(__file__).parent
MODEL_DIR = DIR / 'results'
EPOCHS = 100



class SimpleClassifier(nn.Module):
    def __init__(self, trial, n_classes, embedding_size=768 * 2):
        super().__init__()
        self.rel_output = nn.Linear(embedding_size, n_classes)
        self.dropout = nn.Dropout(trial.suggest_uniform('dropout', 0.0, 0.7))

    def forward(self, x):
        x = self.dropout(x)
        return self.rel_output(x)


class F1Macro(ignite.metrics.Metric):
    def __init__(self, labels, output_transform=lambda x: x,
                 device=None):
        self.labels = labels
        self.precision = Precision(average=False)
        self.recall = Recall(average=False)
        super(F1Macro, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.precision.reset()
        self.recall.reset()

    @reinit__is_reduced
    def update(self, output):
        self.precision.update(output)
        self.recall.update(output)

    @sync_all_reduce("precision", "recall")
    def compute(self):
        precs = self.precision.compute()[self.labels]
        recs = self.recall.compute()[self.labels]
        f1s = (2 * precs * recs / (precs + recs))
        f1s[torch.isnan(f1s)] = 0

        return f1s.mean()


def get_data_loaders(train_batch_size, dev_batch_size):
    return (DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True),
            DataLoader(dev_dataset, batch_size=dev_batch_size, shuffle=False))


def objective(trial: optuna.Trial):
    if not args.disable_wandb:
        wandb.init(project="test_embeddings", reinit=True)
    train_batch_size = 64
    dev_batch_size = 512
    train_loader, dev_loader = get_data_loaders(train_batch_size, dev_batch_size)
    model = SimpleClassifier(trial=trial, n_classes=len(train_dataset.rel_dict),
                             embedding_size=train_dataset.embedding_size)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    optimizer = AdamW(model.parameters(),
                      lr=trial.suggest_loguniform('lr', 3e-6, 3e-3),
                      weight_decay=trial.suggest_uniform('l2', 0, 1)
                      )

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.from_numpy(
            compute_class_weight(class_weight='balanced',
                                 classes=np.unique(train_dataset.y_rels),
                                 y=train_dataset.y_rels.numpy()
                                 )).float().to(device)
    )
    trainer = create_supervised_trainer(model=model, optimizer=optimizer,
                                        device=device, loss_fn=loss_fn)
    pbar = ProgressBar()
    pbar.attach(trainer)
    pos_rel_labels = [v for k, v in train_dataset.rel_dict.items() if k != 'No']
    evaluator = create_supervised_evaluator(model=model, metrics={
        'f1': F1Macro(labels=pos_rel_labels)}, device=device)
    # pbar.attach(evaluator)

    pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, 'f1',
                                                                     trainer)
    pruning_evaluator = create_supervised_evaluator(model=model, metrics={
        'f1': F1Macro(labels=pos_rel_labels)}, device=device)
    pruning_evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    if not args.disable_wandb:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_results(engine):
            pruning_evaluator.run(dev_loader)
            wandb.log(pruning_evaluator.state.metrics)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            wandb.log({'loss': trainer.state.output})

    trainer.run(train_loader, max_epochs=args.epochs)
    evaluator.run(dev_loader)

    return evaluator.state.metrics['f1']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, type=Path)
    parser.add_argument('--dev_data', required=True, type=Path)
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--multiply', action='store_true')
    parser.add_argument('--disable_wandb', action='store_true')
    args = parser.parse_args()
    embedder, embedder_model_path = args.embedder.split(':')

    if embedder == 'BERTEntity':
        embedder = BERTEntityEmbedder(embedder_model_path, multiply=bool(args.multiply))
    elif embedder == "ELMO":
        embedder = MyElmoEmbedder(embedder_model_path)
    else:
        raise ValueError(embedder)

    train_dataset = BioNLPClassificationDataset(df=pd.read_csv(args.train_data),
                                                embedder=embedder)
    dev_dataset = BioNLPClassificationDataset(df=pd.read_csv(args.dev_data),
                                              embedder=embedder,
                                              mod_dict=train_dataset.mod_dict,
                                              rel_dict=train_dataset.rel_dict)
    del embedder
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=args.trials)

    trial = study.best_trial
    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    shutil.rmtree(MODEL_DIR)
