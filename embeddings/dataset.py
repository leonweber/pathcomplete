import numpy as np
import torch
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

    def embed(self, texts, batch_size=8):
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


class BioNLPMatchingDataset(Dataset):
    def __init__(self, df, tokenizer):
        # df = df.sample(100)
        self.input_ids = []
        self.attention_mask = []
        self.e1_starts = []
        self.e2_starts = []
        self.labels = []

        for text, label in zip(df.text, df.labels):
            bert_inputs = tokenizer.encode_plus(text, max_length=512,
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
            self.labels.append(rel)
            self.input_ids.append(bert_inputs['input_ids'])
            self.attention_mask.append(bert_inputs['attention_mask'])

        sort_idx = np.argsort(self.labels)
        self.labels = np.array(self.labels)[sort_idx]
        self.input_ids = torch.from_numpy(np.array(self.input_ids)[sort_idx]).long()
        self.attention_mask = torch.from_numpy(
            np.array(np.array(self.attention_mask)[sort_idx])).long()
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
        attention_mask1 = self.attention_mask[item]
        label = self.labels[item]
        label_idx = self.label_to_idx[label]

        pos_index = np.random.randint(self.label_boundaries[label_idx],
                                      self.label_boundaries[label_idx + 1], 1)[0]

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
        pos_attention_mask = self.attention_mask[pos_index]

        neg_input_ids = self.input_ids[neg_index]
        neg_attention_mask = self.attention_mask[neg_index]

        input = {
            'input_ids': torch.stack([input_ids1, pos_input_ids, neg_input_ids]),
            'attention_mask': torch.stack(
                [attention_mask1, pos_attention_mask, neg_attention_mask]),
            'e1_start': torch.stack([self.e1_starts[item], self.e1_starts[pos_index],
                                     self.e1_starts[neg_index]]),
            'e2_start': torch.stack([self.e2_starts[item], self.e2_starts[pos_index],
                                     self.e2_starts[neg_index]]),
        }

        # return torch.stack([input_ids1, input_ids2]), torch.stack([self.e1_starts[item], self.e1_starts[index]]), \
        #         torch.stack([self.e2_starts[item], self.e2_starts[index]]), torch.tensor(label)
        return input

    def __len__(self):
        return len(self.labels)


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

            bert_inputs = tokenizer.encode_plus(text, max_length=512,
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
