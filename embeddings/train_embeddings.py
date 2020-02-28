import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import pandas as pd
import torch
from pytorch_lightning.callbacks.pt_callbacks import Callback
from pytorch_lightning.logging import MLFlowLogger
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizerFast


def split_label(label):
    if '|' in label:
        fields = label.split('|')
        rel = fields[-1]
        mods = set(fields[:-1])
    else:
        rel = label
        mods = set()

    return rel, mods


class BioNLPMatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.sample(1000)
        self.input_ids = []
        self.e1_starts = []
        self.e2_starts = []
        self.labels = []


        for text, label in zip(df.text, df.labels):
            input_ids = tokenizer.encode(text, max_length=512, pad_to_max_length=True)
            subword_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
            try:
                e1_idx = subword_tokens.index('<e1>')
                e2_idx = subword_tokens.index('<e2>')
            except ValueError:
                continue

            self.e1_starts.append(e1_idx)
            self.e2_starts.append(e2_idx)
            self.labels.append(label)
            self.input_ids.append(input_ids)


        sort_idx = np.argsort(self.labels)
        self.labels = np.array(self.labels)[sort_idx]
        self.input_ids = torch.from_numpy(np.array(self.input_ids)[sort_idx]).long()
        self.e1_starts = torch.from_numpy(np.array(self.e1_starts)[sort_idx]).long()
        self.e2_starts = torch.from_numpy(np.array(self.e2_starts)[sort_idx]).long()

        self.label_boundaries = []
        self.label_to_idx = {}
        for label in np.unique(self.labels):
            self.label_boundaries.append(np.where(label == self.labels)[0][0])
            self.label_to_idx[label] = len(self.label_to_idx)
        self.label_boundaries.append(len(self.labels))

    def __getitem__(self, item):
        input_ids1 = self.input_ids[item]
        label = self.labels[item]
        label_idx = self.label_to_idx[label]

        pos_index = np.random.randint(self.label_boundaries[label_idx], self.label_boundaries[label_idx+1], 1)[0]

        # sample uniformly from examples without the chosen label
        forbidden_start = self.label_boundaries[label_idx]
        forbidden_end = self.label_boundaries[label_idx + 1]

        n_left = forbidden_start
        n_right = len(self.labels) - forbidden_end
        left_prob = (n_left / (n_left + n_right))
        take_left = np.random.uniform(0, 1) < left_prob

        if take_left:
            neg_index = np.random.randint(0, forbidden_start, 1)[0]
        else:
            neg_index = np.random.randint(forbidden_end, len(self.labels), 1)[0]

        pos_input_ids = self.input_ids[pos_index]
        neg_input_ids = self.input_ids[neg_index]

        input = {
            'input_ids': torch.stack([input_ids1, pos_input_ids, neg_input_ids]),
            'e1_start': torch.stack([self.e1_starts[item], self.e1_starts[pos_index], self.e1_starts[neg_index]]),
            'e2_start': torch.stack([self.e2_starts[item], self.e2_starts[pos_index], self.e2_starts[neg_index]]),
        }

        # return torch.stack([input_ids1, input_ids2]), torch.stack([self.e1_starts[item], self.e1_starts[index]]), \
        #         torch.stack([self.e2_starts[item], self.e2_starts[index]]), torch.tensor(label)
        return input

    def __len__(self):
        return len(self.labels)


class SentenceMatcher(pl.LightningModule):
    def __init__(self, dataset, bert_path):
        super().__init__()
        self.data = dataset

        self.bert = BertModel.from_pretrained(bert_path)
        self.output_layer = nn.Linear(768*3*2, 1)
        self.loss_fun = nn.TripletMarginLoss()

    def forward(self, x):
        input_ids = x['input_ids'] # shape == (batch_size, sentences (2), length (512))
        batch_size, sentences, length = input_ids.shape
        e1_start = x['e1_start'] # shape == (batch_size, sentences (2))
        e2_start = x['e2_start'] # shape == (batch_size, sentences (2))
        embs = self.bert(input_ids.view((-1, input_ids.size(2))))[0] # shape == (batch_size * sentences, length (512), 768)
        e1_embs = embs[input_ids.size(0), e1_start.view(-1)] # shape == (batch * sentences, 768)
        e2_embs = embs[input_ids.size(0), e2_start.view(-1)] # shape == (batch * sentences, 768)
        embs = torch.cat([e1_embs, e2_embs], dim=1).reshape((batch_size, sentences, -1)) # shape == (batch, sentences, 768*2)

        return embs[:, 0, ...], embs[:, 1, ...], embs[:, 2, ...]



    def training_step(self, batch, batch_idx):
        anchor, pos, neg = self(batch)
        loss = self.loss_fun(anchor, pos, neg)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    # def validation_step(self, batch, batch_idx):
    #     x, y_rel = batch
    #     y_rel_pred = self(x)
    #     loss = self.rel_loss_fun(y_rel_pred, y_rel)
    #
    #     return {'val_loss': loss, 'y_rel_pred': y_rel_pred, 'y_rel_true': y_rel}
    #
    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     y_rel_pred = torch.cat([x['y_rel_pred'] for x in outputs])
    #     y_rel_true = torch.cat([x['y_rel_true'] for x in outputs])
    #
    #     f1 = f1_score(y_rel_true.cpu(), y_rel_pred.argmax(dim=1).cpu)
    #     log = {'val_loss': avg_loss, 'f1': f1}
    #     return {'avg_val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.data, shuffle=True, batch_size=2)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bert', required=True)
    args = parser.parse_args()
    logger = MLFlowLogger('embeddings')

    tokenizer = BertTokenizerFast.from_pretrained(args.bert)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})
    dataset = BioNLPMatchingDataset(pd.read_csv(args.data), tokenizer=tokenizer)
    model = SentenceMatcher(dataset, args.bert)
    trainer = pl.Trainer(early_stop_callback=None, logger=logger, gpus=1, max_epochs=1, checkpoint_callback=None)
    trainer.fit(model)
    pass
