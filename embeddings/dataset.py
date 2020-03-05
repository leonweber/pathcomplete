import logging

import numpy as np
import torch
from allennlp.commands.elmo import ElmoEmbedder
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast


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


class BERTEntityEmbedder:
    def __init__(self, bert, multiply=False):
        self.bert = nn.DataParallel(BertModel.from_pretrained(bert))
        self.bert.cuda()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})
        self.multiply = multiply
        self.embedding_size = 768 * 2

    def embed(self, texts, batch_size=8):
        with torch.no_grad():
            embs = []
            batch_size = batch_size * torch.cuda.device_count()
            for chunk in tqdm(list(chunks(texts, batch_size))):
                input_ids = []
                attention_mask = []
                e1_starts = []
                e2_starts = []
                for text in chunk:
                    bert_input = self.tokenizer.encode_plus(text, max_length=512,
                                                            pad_to_max_length=True,
                                                            add_special_tokens=True)
                    subword_tokens = self.tokenizer.convert_ids_to_tokens(
                        bert_input['input_ids'],
                        skip_special_tokens=False)
                    try:
                        e1_idx = subword_tokens.index('<e1>')
                        e2_idx = subword_tokens.index('<e2>')
                    except ValueError:
                        embs.append(torch.zeros(1, 768 * 2))
                        continue

                    e1_starts.append(e1_idx)
                    e2_starts.append(e2_idx)

                    input_ids.append(bert_input['input_ids'])
                    attention_mask.append(bert_input['attention_mask'])
                input_ids = torch.tensor(input_ids).long().cuda()
                attention_mask = torch.tensor(attention_mask).long().cuda()
                chunk_embs = self.bert(input_ids=input_ids,
                                       attention_mask=attention_mask)[
                    0].cpu().detach().numpy()

                e1_embs = torch.from_numpy(
                    chunk_embs[np.arange(len(chunk_embs)), np.array(e1_starts), ...])
                e2_embs = torch.from_numpy(
                    chunk_embs[np.arange(len(chunk_embs)), np.array(e2_starts), ...])

                if self.multiply:
                    embs.append(torch.cat([e1_embs, e2_embs, e1_embs * e2_embs], dim=1))
                else:
                    embs.append(torch.cat([e1_embs, e2_embs], dim=1))
            embs = torch.cat(embs, dim=0)

            return embs


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class BERTCLSEmbedder:
    def __init__(self, bert):
        self.bert = BertModel.from_pretrained(bert)
        self.bert.cuda()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})

    def embed(self, texts, batch_size=8):
        embs = []
        for chunk in tqdm(list(chunks(texts, batch_size))):
            input_ids = []
            for text in chunk:
                input_id = self.tokenizer.encode(text, max_length=512,
                                                 pad_to_max_length=True)
                input_ids.append(input_id)
            input_ids = torch.tensor(input_ids).long().cuda()
            chunk_embs = self.bert(input_ids)[0].cpu().detach().numpy()
            embs.append(chunk_embs[:, 0, ...])

        embs = torch.cat(embs, dim=0)

        return embs


class BERTTokenizerEmbedder:
    def __init__(self, tokenizer):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<e1>', '</e1>', '<e2>', '</e2>']})

    def embed(self, texts):
        input_ids = []
        for text in texts:
            input_id = self.tokenizer.encode_plus(text, max_length=512,
                                                  pad_to_max_length=True,
                                                  add_special_tokens=True)
            input_ids.append(input_id)

        return input_ids


class MyElmoEmbedder:
    def __init__(self, elmo):
        self.elmo = ElmoEmbedder(
            options_file=elmo + "/biomed_elmo_options.json",
            weight_file=elmo + "/biomed_elmo_weights.hdf5",
            cuda_device=0)
        self.embedding_size = 3 * 512 * 4

    def embed(self, texts):
        all_tokens = []
        for text in texts:
            tokens = text.split()
            all_tokens.append(text.replace("<e1>", "").replace("</e1>", "")
                              .replace("</e1>", "").replace("</e1>", "").split())

        embs = list(self.elmo.embed_sentences(sentences=all_tokens))
        final_embs = []

        for text, emb in zip(texts, embs):
            tokens = text.split()
            e1_start = [i for i, t in enumerate(tokens) if '<e1>' in t][0]
            e1_end = [i for i, t in enumerate(tokens) if '</e1>' in t][0]
            e2_start = [i for i, t in enumerate(tokens) if '<e2>' in t][0]
            e2_end = [i for i, t in enumerate(tokens) if '</e2>' in t][0]

            e1_emb_fwd = emb[:, e1_start, :512].ravel()
            e1_emb_bwd = emb[:, e1_end, 512:].ravel()
            e2_emb_fwd = emb[:, e2_start, :512].ravel()
            e2_emb_bwd = emb[:, e2_end, 512:].ravel()

            final_embs.append(torch.from_numpy(
                np.hstack([e1_emb_fwd, e1_emb_bwd, e2_emb_fwd, e2_emb_bwd])))

        return torch.stack(final_embs, dim=0)


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
            bert_inputs = tokenizer.encode_plus(text, max_length=328,
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

        self.curr_label_to_idx = None
        self.curr_labels = None
        self.curr_input_ids = None
        self.curr_attention_mask = None
        self.curr_e1_starts = None
        self.curr_e2_starts = None

        self.resample_indices()

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

            bert_inputs = tokenizer.encode_plus(text, max_length=368,
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
