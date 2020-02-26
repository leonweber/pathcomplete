import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import WandbLogger
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast

from util.torch_util import multilabels_to_onehot
from .model import SimpleClassifier



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class BERTEntityEmbedder:
    def __init__(self, bert):
        self.bert = BertModel.from_pretrained(bert)
        self.bert.cuda()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})

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

            embs.append(torch.cat([e1_embs, e2_embs], dim=1))
        embs = torch.cat(embs, dim=0)


        return embs


class BioNLPClassificationDataset(Dataset):
    def __init__(self, path, embedder, rel_dict=None, mod_dict=None):
        self.df = pd.read_csv(path)
        self.X = embedder.embed(self.df.text.tolist())
        self.rel_dict = rel_dict or {}
        self.mod_dict = mod_dict or {}
        self.y_rels, self.y_mods = self.split_labels()

    def split_labels(self):
        all_rels = []
        all_mods = []
        for label in self.df.labels:
            if '|' in label:
                fields = label.split('|')
                rel = fields[-1]
                mods = fields[:-1]
            else:
                rel = label
                mods = []

            if rel not in self.rel_dict:
                self.rel_dict[rel] = len(self.rel_dict)
            all_rels.append(self.rel_dict[rel])

            mod_vals = []
            for mod in mods:
                if mod not in self.mod_dict:
                    self.mod_dict[mod] = len(self.mod_dict)
                mod_vals.append(self.mod_dict[mod])
            all_mods.append(mod_vals)

        return torch.tensor(all_rels), torch.from_numpy(multilabels_to_onehot(all_mods, len(self.mod_dict)))

    def __getitem__(self, item):
        return self.X[item], self.y_rels[item], self.y_mods[item]

    def __len__(self):
        return len(self.X)


class EmbeddingTester(pl.LightningModule):
    def __init__(self, n_rel_types, n_mod_types, train_dataset, dev_dataset, embedding_size=768*2, rel_loss_weight=0.5):
        super().__init__()
        self.train_data = train_dataset
        self.dev_data = dev_dataset
        self.model = SimpleClassifier(n_rel_types=n_rel_types, n_mod_types=n_mod_types,
                                      embedding_size=embedding_size)
        self.rel_loss_fun = nn.CrossEntropyLoss()
        self.mod_loss_fun = nn.BCEWithLogitsLoss()
        self.rel_loss_weight = rel_loss_weight

    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y_rel, y_mod = batch
        y_rel_pred, y_mod_pred = self.model.forward(x)
        rel_loss = self.rel_loss_fun(y_rel_pred, y_rel)
        mod_loss = self.mod_loss_fun(y_mod_pred, y_mod)
        loss = rel_loss * self.rel_loss_weight + mod_loss * (1-self.rel_loss_weight)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y_rel, y_mod = batch
        y_rel_pred, y_mod_pred = self.model.forward(x)
        rel_loss = self.rel_loss_fun(y_rel_pred, y_rel)
        mod_loss = self.mod_loss_fun(y_mod_pred, y_mod)

        loss = rel_loss * self.rel_loss_weight + mod_loss * (1-self.rel_loss_weight)
        return {'val_loss': loss, 'y_rel_pred': y_rel_pred, 'y_mod_pred': y_mod_pred, 'y_rel_true': y_rel, 'y_mod_true': y_mod}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y_rel_pred = torch.cat([x['y_rel_pred'] for x in outputs])
        y_rel_true = torch.cat([x['y_rel_true'] for x in outputs])
        y_mod_pred = torch.cat([x['y_mod_pred'] for x in outputs])
        y_mod_true = torch.cat([x['y_mod_true'] for x in outputs])

        f1_rel = f1_score(y_rel_true, y_rel_pred.argmax(dim=1), average='micro')
        f1_mod = f1_score(y_mod_true.view(-1), (y_mod_pred > 0).view(-1))
        log = {'val_loss': avg_loss, 'f1_mod': f1_mod, 'f1_rel': f1_rel}
        return {'avg_val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=64)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dev_data, shuffle=False, batch_size=64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', '-t', required=True, type=Path)
    parser.add_argument('--dev_data', '-d', required=True, type=Path)
    parser.add_argument('--embedder', '-e', required=True)

    args = parser.parse_args()

    embedder, embedder_model_path = args.embedder.split(':')

    if embedder == 'BERTEntity':
        embedder = BERTEntityEmbedder(embedder_model_path)

    train_dataset = BioNLPClassificationDataset(path=args.train_data, embedder=embedder)
    dev_dataset = BioNLPClassificationDataset(path=args.dev_data, embedder=embedder, mod_dict=train_dataset.mod_dict,
                                              rel_dict=train_dataset.rel_dict)

    early_stopping = EarlyStopping(monitor='avg_val_loss', patience=10)
    # logger = WandbLogger(project='embeddings')
    system = EmbeddingTester(n_rel_types=len(train_dataset.rel_dict), n_mod_types=len(train_dataset.mod_dict),
                             train_dataset=train_dataset, dev_dataset=dev_dataset)
    trainer = pl.Trainer(early_stop_callback=early_stopping)
    trainer.fit(system)





