from bisect import bisect_right, bisect_left

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch import nn
from tqdm import tqdm
from transformers import BertModel, PreTrainedModel, BertConfig, BertTokenizerFast
from transformers.modeling_bert import BertOnlyMLMHead, BertPreTrainedModel

from .dataset import chunks




class BertEntity(BertPreTrainedModel):
    def __init__(self, config, n_classes):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768*2, n_classes)
        self.n_classes = n_classes
        self.init_weights()

    def forward(self, x):
        input_ids = x['input_ids']  # shape == (batch_size, sentences (3), length (512))
        batch_size, length = input_ids.shape[:2]
        attention_mask = x['attention_mask']
        e1_start = x['e1_start']  # shape == (batch_size, sentences (3))
        e2_start = x['e2_start']  # shape == (batch_size, sentences (3))
        embs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        e1_embs = embs[torch.arange(batch_size), e1_start]
        e2_embs = embs[torch.arange(batch_size), e2_start]
        embs = torch.cat([e1_embs, e2_embs], dim=1)

        embs = self.dropout(embs)

        return self.classifier(embs)



class SentenceMatcher(BertPreTrainedModel):
    def __init__(self, config, e1_start_id, e2_start_id):
        super().__init__(config)

        self.bert = BertModel(config)
        self.mlm_head = BertOnlyMLMHead(config)
        self.e1_start_id = e1_start_id
        self.e2_start_id = e2_start_id

        self.init_weights()


    def forward(self, x):
        input_ids = x['input_ids']  # shape == (batch_size, sentences (3), length (512))
        self.e1_start_id = self.e1_start_id.to(input_ids.device)
        self.e2_start_id = self.e2_start_id.to(input_ids.device)
        e1_starts = input_ids.eq(self.e1_start_id)
        e2_starts = input_ids.eq(self.e2_start_id)
        attention_mask = x['attention_mask']
        batch_size, sentences, length = input_ids.shape
        embs = self.bert(input_ids=input_ids.view((-1, length)),
                         attention_mask=attention_mask.view((-1, length)))[
            0]  # shape == (batch_size * sentences, length (513), 768)

        mlm_logits = self.mlm_head(embs)

        embs = embs.view((batch_size, sentences, length, -1))
        embs = torch.cat([embs[e1_starts], embs[e2_starts]], dim=1).reshape(
            (batch_size, sentences, -1))  # shape == (batch, sentences, 768*2)


        return embs, mlm_logits


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
            for chunk in list(chunks(texts, batch_size)):
                input_ids = []
                attention_mask = []
                e1_starts = []
                e2_starts = []
                for text in chunk:
                    bert_input = self.tokenizer.encode_plus(text, max_length=256,
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
                if not input_ids:
                    return
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


class W2VEmbedder:
    def __init__(self, embeddings):
        self.embeddings = KeyedVectors.load_word2vec_format(embeddings, binary=True)
        self.embedding_size = 200 * 2

    def embed(self, texts):
        embedded_texts = []
        for text in texts:
            e1 = text[text.index('<e1>')+4:text.index('</e1>')]
            e2 = text[text.index('<e2>')+4:text.index('</e2>')]

            e1_emb = []
            for t in e1.lower().split():
                if t in self.embeddings:
                    e1_emb.append(self.embeddings[t])
            if e1_emb:
                e1_emb = np.mean(e1_emb, axis=0)
            else:
                e1_emb = np.zeros(self.embedding_size//2)

            e2_emb = []
            for t in e2.lower().split():
                if t in self.embeddings:
                    e2_emb.append(self.embeddings[t])
            if e2_emb:
                e2_emb = np.mean(e2_emb, axis=0)
            else:
                e2_emb = np.zeros(self.embedding_size//2)

            emb = np.hstack([e1_emb, e2_emb])
            embedded_texts.append(emb)

        return torch.tensor(embedded_texts).float()


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
                input_id = self.tokenizer.encode(text, max_length=256,
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
            input_id = self.tokenizer.encode_plus(text, max_length=256,
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