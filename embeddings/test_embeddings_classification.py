import argparse
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import MLFlowLogger, LightningLoggerBase
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification
import optuna
from optuna.integration import PyTorchLightningPruningCallback



from util.torch_util import multilabels_to_onehot

DIR = Path(__file__).parent
MODEL_DIR = DIR / 'results'

class DictLogger(LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self, version, offline_logger):
        super(DictLogger, self).__init__()
        self.metrics = []
        self._version = version
        self.offline_logger = offline_logger

    def log_metrics(self, metric, step=None):
        self.metrics.append(metric)
        self.offline_logger.log_metrics(metric, step)

    def log_hyperparams(self, params):
        self.offline_logger.log_hyperparams(params)

    def finalize(self, status="FINISHED"):
        self.offline_logger.finalize(status)

    @property
    def name(self):
        return self.offline_logger.name

    @property
    def run_id(self):
        return self.offline_logger.run_id

    @property
    def version(self):
        return self._version



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(embedder_model_path)

    def forward(self, x):
        return self.bert(x)[0]



class SimpleClassifier(nn.Module):
    def __init__(self, trial):
        super().__init__()
        self.rel_output = nn.Linear(trial.embedding_size, trial.n_rel_types)
        self.dropout = nn.Dropout(trial.suggest_uniform('dropout', 0.0, 0.7))

    def forward(self, x):
        x = self.dropout(x)
        return self.rel_output(x)


class BERTEntityEmbedder:
    def __init__(self, bert, multiply=False):
        self.bert = BertModel.from_pretrained(bert)
        self.bert.cuda()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})
        self.multiply = multiply

    def embed(self, texts, batch_size=8):
        embs = []
        for chunk in tqdm(list(chunks(texts, batch_size))):
            input_ids = []
            e1_starts = []
            e2_starts = []
            for text in chunk:
                input_id = self.tokenizer.encode(text, max_length=512, pad_to_max_length=True)
                subword_tokens = self.tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=False)
                try:
                    e1_idx = subword_tokens.index('<e1>')
                    e2_idx = subword_tokens.index('<e2>')
                except ValueError:
                    embs.append(torch.zeros(1, 768*2))
                    continue

                e1_starts.append(e1_idx)
                e2_starts.append(e2_idx)

                input_ids.append(input_id)
            input_ids = torch.tensor(input_ids).long().cuda()
            chunk_embs = self.bert(input_ids)[0].cpu().detach().numpy()

            e1_embs = torch.from_numpy(chunk_embs[np.arange(len(chunk_embs)), np.array(e1_starts), ...])
            e2_embs = torch.from_numpy(chunk_embs[np.arange(len(chunk_embs)), np.array(e2_starts), ...])

            if self.multiply:
                embs.append(torch.cat([e1_embs, e2_embs, e1_embs * e2_embs], dim=1))
            else:
                embs.append(torch.cat([e1_embs, e2_embs], dim=1))
        embs = torch.cat(embs, dim=0)


        return embs


class BERTCLSEmbedder:
    def __init__(self, bert):
        self.bert = BertModel.from_pretrained(bert)
        self.bert.cuda()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})

    def embed(self, texts, batch_size=8):
        embs = []
        for chunk in tqdm(list(chunks(texts, batch_size))):
            input_ids = []
            for text in chunk:
                input_id = self.tokenizer.encode(text, max_length=512, pad_to_max_length=True)
                input_ids.append(input_id)
            input_ids = torch.tensor(input_ids).long().cuda()
            chunk_embs = self.bert(input_ids)[0].cpu().detach().numpy()
            embs.append(chunk_embs[:, 0, ...])

        embs = torch.cat(embs, dim=0)

        return embs


class BERTTokenizerEmbedder:
    def __init__(self, tokenizer):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})

    def embed(self, texts):
        input_ids = []
        for text in texts:
            input_id = self.tokenizer.encode(text, max_length=512, pad_to_max_length=True)
            input_ids.append(input_id)

        return torch.tensor(input_ids).long()


class BioNLPClassificationDataset(Dataset):
    def __init__(self, df, embedder, rel_dict=None, mod_dict=None, modified_is_negative=True):
        self.df = df
        self.modified_is_negative = modified_is_negative
        if embedder:
            self.X = embedder.embed(self.df.text.tolist())
        self.rel_dict = rel_dict or {}
        self.mod_dict = mod_dict or {}
        self.y_rels = self.split_labels()

    def split_labels(self):
        all_rels = []
        for label in self.df.labels:
            if '|' in label:
                fields = label.split('|')
                rel = fields[-1]
                mods = fields[:-1]
            else:
                rel = label
                mods = []

            if self.modified_is_negative and mods:
                rel = 'No'

            if rel not in self.rel_dict:
                self.rel_dict[rel] = len(self.rel_dict)
            all_rels.append(self.rel_dict[rel])

        return torch.tensor(all_rels)

    def __getitem__(self, item):
        return self.X[item], self.y_rels[item]

    def __len__(self):
        return len(self.X)


class EmbeddingTester(pl.LightningModule):
    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.train_data: BioNLPClassificationDataset = trial.train_dataset
        self.dev_data: BioNLPClassificationDataset = trial.dev_dataset

        self._model = ModelType(trial)
        self.rel_loss_fun = nn.CrossEntropyLoss(
            weight=torch.from_numpy(compute_class_weight(class_weight='balanced',
                                        classes=np.unique(self.train_data.y_rels),
                                        y=self.train_data.y_rels.numpy()
                                        )).float())

        self.l2_reg = trial.suggest_uniform('l2', 0.0, 1.0)

    def forward(self, x):
        return self._model(x)


    def training_step(self, batch, batch_idx):
        x, y_rel = batch
        y_rel_pred = self(x)
        loss = self.rel_loss_fun(y_rel_pred, y_rel)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y_rel = batch
        y_rel_pred = self(x)
        loss = self.rel_loss_fun(y_rel_pred, y_rel)

        return {'val_loss': loss, 'y_rel_pred': y_rel_pred, 'y_rel_true': y_rel}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y_rel_pred = torch.cat([x['y_rel_pred'] for x in outputs])
        y_rel_true = torch.cat([x['y_rel_true'] for x in outputs])

        pos_rel_labels = [v for k, v in self.train_data.rel_dict.items() if k != 'No']
        f1 = f1_score(y_rel_true.cpu(), y_rel_pred.argmax(dim=1).cpu(), average='macro', labels=pos_rel_labels)
        log = {'val_loss': avg_loss, 'f1': f1}
        return {'avg_val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=self.l2_reg)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=64)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dev_data, shuffle=False, batch_size=64)


def objective(trial: optuna.Trial):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'trial_{}'.format(trial.number)), monitor='accuracy')
    early_stopping = PyTorchLightningPruningCallback(trial, monitor='f1')
    logger = DictLogger(trial.number, MLFlowLogger('embeddings'))
    trial.n_rel_types = len(train_dataset.rel_dict)
    trial.train_dataset = train_dataset
    trial.dev_dataset = dev_dataset
    if args.multiply:
        trial.embedding_size = 768 * 3
    else:
        trial.embedding_size = 768 * 2

    system = EmbeddingTester(trial)
    trainer = pl.Trainer(early_stop_callback=early_stopping, logger=logger, gpus=1,
                         checkpoint_callback=checkpoint_callback, max_epochs=100)
    trainer.fit(system)

    return logger.metrics[-1]['f1']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', required=True, type=Path)
    parser.add_argument('--dev_data', required=True, type=Path)
    parser.add_argument('--embedder', required=True)
    parser.add_argument('--multiply', action='store_true')
    args = parser.parse_args()
    embedder, embedder_model_path = args.embedder.split(':')

    if embedder == 'BERTEntity':
        embedder = BERTEntityEmbedder(embedder_model_path, multiply=bool(args.multiply))
        ModelType = SimpleClassifier
    elif embedder == 'BERTTokenizer':
        embedder = BERTTokenizerEmbedder(embedder_model_path)
        ModelType = BERT

    train_dataset = BioNLPClassificationDataset(df=pd.read_csv(args.train_data), embedder=embedder)
    dev_dataset = BioNLPClassificationDataset(df=pd.read_csv(args.dev_data), embedder=embedder, mod_dict=train_dataset.mod_dict,
                                              rel_dict=train_dataset.rel_dict)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100)

    trial = study.best_trial
    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    shutil.rmtree(MODEL_DIR)




