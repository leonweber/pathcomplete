import logging

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.utils.data import Dataset


class BioNLPClassificationDataset(Dataset):
    def __init__(self, df, embedder, rel_dict=None, mod_dict=None,
                 modified_is_negative=True):
        self.df = df
        # self.df = self.df.sample(1000)
        self.modified_is_negative = modified_is_negative
        if embedder:
            self.X = embedder.embed(self.df.text.tolist())
            self.embedding_size = embedder.embedding_size
        self.rel_dict = rel_dict or {}
        self.mod_dict = mod_dict or {}
        self.y_rels = self.split_labels()

        # self.X = self.X[self.y_rels != 0]
        # self.y_rels = self.y_rels[self.y_rels != 0]

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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class BioNLPMatchingDataset(Dataset):
    def __init__(self, df, tokenizer, test=False):
        if test:
            df = df.sample(1000)
        self.input_ids = []
        self.attention_mask = []
        self.e1_starts = []
        self.e2_starts = []
        self.label_dict = {}
        self.labels = []
        self.curr_dist_matrix = None

        for text, label in zip(df.text, df.labels):
            bert_inputs = tokenizer.encode_plus(text, max_length=256,
                                                pad_to_max_length=True,
                                                add_special_tokens=True)
            subword_tokens = tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'],
                                                             skip_special_tokens=False)

            if '|' in label:
                fields = label.split('|')
                rel = fields[-1]
                mods = fields[:-1]
            else:
                rel = label
                mods = []

            if mods:
                rel = 'No'

            try:
                e1_idx = subword_tokens.index('<e1>')
                e2_idx = subword_tokens.index('<e2>')
            except ValueError:
                continue

            self.e1_starts.append(e1_idx)
            self.e2_starts.append(e2_idx)

            if rel not in self.label_dict:
                self.label_dict[rel] = len(self.label_dict)

            self.labels.append(self.label_dict[rel])
            self.input_ids.append(bert_inputs['input_ids'])
            self.attention_mask.append(bert_inputs['attention_mask'])

        sort_idx = np.argsort(self.labels)
        self.labels = np.array(self.labels)[sort_idx]
        self.input_ids = torch.from_numpy(np.array(self.input_ids)[sort_idx]).long()
        self.attention_mask = torch.from_numpy(
            np.array(np.array(self.attention_mask)[sort_idx])).long()
        self.e1_starts = torch.from_numpy(np.array(self.e1_starts)[sort_idx]).long()
        self.e2_starts = torch.from_numpy(np.array(self.e2_starts)[sort_idx]).long()

        self.label_dict_inv = {v: k for k, v in self.label_dict.items()}

        self.curr_labels = self.labels
        self.curr_input_ids = self.input_ids
        self.curr_attention_mask = self.attention_mask
        self.curr_e1_starts = self.e1_starts
        self.curr_e2_starts = self.e2_starts

    def __getitem__(self, idx):
        label = self.curr_labels[idx]
        logging.debug('label: ' + self.label_dict_inv[label])

        pos_indices = np.where(self.curr_labels == label)[0]
        neg_indices = np.where(self.curr_labels != label)[0]

        if self.curr_dist_matrix is None:
            pos_idx = np.random.choice(pos_indices, size=1, replace=False)[0]
            neg_idx = np.random.choice(neg_indices, size=1, replace=False)[0]
        else:
            pos_idx = pos_indices[self.curr_dist_matrix[idx, pos_indices].argmax()]
            neg_idx = neg_indices[self.curr_dist_matrix[idx, neg_indices].argmin()]

        input = {
            'input_ids': torch.stack([self.curr_input_ids[idx],
                                      self.curr_input_ids[pos_idx],
                                      self.curr_input_ids[neg_idx]]),
            'attention_mask': torch.stack([self.curr_attention_mask[idx],
                                           self.curr_attention_mask[pos_idx],
                                           self.curr_attention_mask[neg_idx]]),
            'e1_start': torch.stack(
                [self.curr_e1_starts[idx], self.curr_e1_starts[pos_idx],
                 self.curr_e1_starts[neg_idx]]),
            'e2_start': torch.stack(
                [self.curr_e2_starts[idx], self.curr_e2_starts[pos_idx],
                 self.curr_e2_starts[neg_idx]]),
        }

        return input

    def __len__(self):
        return len(self.curr_labels)

    def resample_indices(self):
        label_count = np.bincount(self.labels)
        min = label_count.min()
        max_count = 10 * min
        curr_indices = []

        for label, count in enumerate(label_count):
            label_indices = np.where(self.labels == label)[0]
            if count > max_count:
                sampled_label_indices = np.random.choice(label_indices, size=max_count,
                                                         replace=False)
                curr_indices.extend(sampled_label_indices)
            else:
                curr_indices.extend(label_indices)
        curr_indices = torch.from_numpy(np.array(curr_indices))

        self.curr_labels = self.labels[curr_indices.numpy()]
        self.curr_input_ids = self.input_ids[curr_indices]
        self.curr_attention_mask = self.attention_mask[curr_indices]
        self.curr_e1_starts = self.e1_starts[curr_indices]
        self.curr_e2_starts = self.e2_starts[curr_indices]

    def update_dist_matrix(self, model):
        model = nn.DataParallel(model)
        embs = []
        batch_size = 2 * torch.cuda.device_count()
        for batch in chunks(np.arange(len(self.curr_input_ids)), batch_size):
            with torch.no_grad():
                e1_starts = self.curr_e1_starts[batch]
                e2_starts = self.curr_e2_starts[batch]
                x = model(input_ids=self.curr_input_ids[batch].cuda(),
                          attention_mask=self.attention_mask[
                              batch].cuda())[0]
                e1_emb = x[torch.arange(len(batch)), e1_starts]
                e2_emb = x[torch.arange(len(batch)), e2_starts]
                emb = torch.cat([e1_emb, e2_emb], dim=1)
                embs.append(emb.cpu().detach().numpy())

        embs = np.vstack(embs)
        self.curr_dist_matrix = pairwise_distances(embs)


class BioNLPBertClassDataset(Dataset):
    def __init__(self, df, tokenizer, rel_vocab=None):

        # df = df.sample(100)
        self.input_ids = []
        self.attention_mask = []
        self.e1_starts = []
        self.e2_starts = []
        self.labels = []
        self.rel_vocab = rel_vocab or {}

        for text, label in zip(df.text, df.labels):
            if '|' in label:
                fields = label.split('|')
                rel = fields[-1]
                mods = fields[:-1]
            else:
                rel = label
                mods = []

            if mods:
                rel = 'No'

            if rel not in self.rel_vocab:
                self.rel_vocab[rel] = len(self.rel_vocab)

            bert_inputs = tokenizer.encode_plus(text, max_length=256,
                                                pad_to_max_length=True,
                                                add_special_tokens=True)
            subword_tokens = tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'],
                                                             skip_special_tokens=False)
            try:
                e1_idx = subword_tokens.index('<e1>')
                e2_idx = subword_tokens.index('<e2>')
            except ValueError:
                continue

            self.e1_starts.append(e1_idx)
            self.e2_starts.append(e2_idx)
            self.labels.append(self.rel_vocab[rel])
            self.input_ids.append(bert_inputs['input_ids'])
            self.attention_mask.append(bert_inputs['attention_mask'])

        self.input_ids = torch.tensor(self.input_ids).long()
        self.attention_mask = torch.tensor(self.attention_mask).long()
        self.e1_starts = torch.tensor(self.e1_starts).long()
        self.e2_starts = torch.tensor(self.e2_starts).long()
        self.labels = torch.tensor(self.labels).long()

    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        attention_mask = self.attention_mask[item]
        label = self.labels[item]

        input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'e1_start': self.e1_starts[item],
            'e2_start': self.e2_starts[item],
        }

        return input, label

    def __len__(self):
        return len(self.labels)


class MTBDataset(Dataset):
    def __init__(self, path, tokenizer, test=False):
        self.input_ids1 = []
        self.attention_mask1 = []

        self.input_ids2 = []
        self.attention_mask2 = []

        self.labels = []
        e1_id = tokenizer.convert_tokens_to_ids('<e1>')
        e2_id = tokenizer.convert_tokens_to_ids('<e2>')

        with open(path) as f:
            for line in f:
                fields = line.strip().split('\t')
                if not fields:
                    continue
                text1, text2, label = fields
                bert_inputs1 = tokenizer.encode_plus(text1, max_length=256,
                                                     pad_to_max_length=True,
                                                     add_special_tokens=True)

                bert_inputs2 = tokenizer.encode_plus(text2, max_length=256,
                                                     pad_to_max_length=True,
                                                     add_special_tokens=True)

                if not (e1_id in bert_inputs1['input_ids'] and
                        e2_id in bert_inputs1['input_ids'] and
                        e1_id in bert_inputs2['input_ids'] and
                        e2_id in bert_inputs2['input_ids']):
                    continue  # make sure <e1> and <e2> tokens are present in both truncated texts

                self.input_ids1.append(bert_inputs1['input_ids'])
                self.attention_mask1.append(bert_inputs1['attention_mask'])

                self.input_ids2.append(bert_inputs2['input_ids'])
                self.attention_mask2.append(bert_inputs2['attention_mask'])

                self.labels.append(int(label))

        self.input_ids1 = torch.tensor(self.input_ids1).long()
        self.input_ids2 = torch.tensor(self.input_ids2).long()
        self.attention_mask1 = torch.tensor(self.attention_mask1).long()
        self.attention_mask2 = torch.tensor(self.attention_mask2).long()
        self.labels = torch.tensor(self.labels).float()

    def __getitem__(self, idx):
        input = {
            'input_ids': torch.stack([self.input_ids1[idx],
                                      self.input_ids2[idx]]),
            'attention_mask': torch.stack([self.attention_mask1[idx],
                                           self.attention_mask2[idx]])
        }

        return input, self.labels[idx]

    def __len__(self):
        return len(self.input_ids1)
