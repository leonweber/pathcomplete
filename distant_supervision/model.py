import math
from copy import deepcopy

import transformers
import numpy as np
from torch import nn
import torch
from torch.nn import DataParallel
from transformers import BertPreTrainedModel


def aggregate_provenance_predictions(alphas, pmids):
    pmid_predictions = {}
    for alpha, pmid in zip(alphas, pmids):
        pmid = pmid.item()
        if pmid not in pmid_predictions:
            pmid_predictions[pmid] = alpha
        else:
            pmid_predictions[pmid] += alpha

    return pmid_predictions


class BagEmbedder(nn.Module):
    def __init__(self, bert, args):
        super(BagEmbedder, self).__init__()
        self.pretrained_classifier_bert = transformers.BertForSequenceClassification.from_pretrained(bert, output_hidden_states=True)
        self.bert = DataParallel(self.pretrained_classifier_bert.bert)
        self._init_weights = self.pretrained_classifier_bert._init_weights
        self.attention_layer = self.pretrained_classifier_bert.classifier

    def forward(self, token_ids, attention_masks, entity_pos, **kwargs):
        outputs = self.bert(token_ids, attention_mask=attention_masks)
        x = outputs[1]
        x = self.pretrained_classifier_bert.dropout(x)
        unnorm_alphas = self.attention_layer(x) # class 0 is positive :/

        alphas = torch.softmax(unnorm_alphas, dim=1)[:, 0]
        # alphas = torch.sigmoid(unnorm_alphas)
        x = (x * alphas.unsqueeze(1))
        x = torch.max(x, dim=0)[0]
        # alphas = torch.zeros(5, 1)
        # x = torch.max(x, dim=0)[0]

        meta = {'alphas_hist': np.histogram(alphas.detach().cpu().numpy()), 'alphas': unnorm_alphas}

        return x, meta


class BagOnly(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.bag_embedder = BagEmbedder(bert, args)
        self.ff_output = nn.Linear(768, args.n_classes)
        self.no_mentions_emb = nn.Embedding(1, 768)
        self._init_weights = self.bag_embedder._init_weights

        self._init_weights(self.ff_output)
        self._init_weights(self.no_mentions_emb)

    def forward(self, token_ids, attention_masks, entity_pos, has_mentions, **kwargs):
        meta = {}

        if has_mentions.sum() > 0:
            x, m = self.bag_embedder(token_ids, attention_masks, entity_pos)
            meta.update(m)
        else:
            x = self.no_mentions_emb(torch.tensor(0).to(token_ids.device))
        x = self.ff_output(x)

        return x, meta

class Complex(nn.Module):
    def __init__(self, bert, args):
        super().__init__()
        self.tensor_emb_size = args.tensor_emb_size
        self.entity_re_embedding = nn.Embedding(args.n_entities, self.tensor_emb_size)
        self.entity_im_embedding = nn.Embedding(args.n_entities, self.tensor_emb_size)
        self.r_re_embedding = nn.Embedding(args.n_classes, self.tensor_emb_size)
        self.r_im_embedding = nn.Embedding(args.n_classes, self.tensor_emb_size)
        self.bag_embedder = BagEmbedder(bert, args)
        self.no_mentions_emb = nn.Parameter(torch.zeros(768).uniform_(-0.02, 0.02))
        self.ff_gate = nn.Sequential(
            nn.Linear(2 * self.tensor_emb_size, 1),
            nn.Sigmoid()
        )
        self.bag_downprojection = nn.Sequential(
            nn.Linear(768, self.tensor_emb_size),
            nn.ReLU()
        )

    def forward(self, entity_ids, entity_pos, token_ids, attention_masks, has_mentions, **kwargs):
        e1_re = self.entity_re_embedding(entity_ids[0])
        e1_im = self.entity_im_embedding(entity_ids[0])
        e2_re = self.entity_re_embedding(entity_ids[1])
        e2_im = self.entity_im_embedding(entity_ids[1])
        r_re = self.r_re_embedding.weight
        r_im = self.r_im_embedding.weight

        es1 = e1_re * e2_re
        es2 = e1_im * e2_im
        es3 = e1_re * e2_im
        es4 = e1_im * e2_re

        meta = {}

        if has_mentions.sum().item() > 0:
            bag_emb, m = self.bag_embedder(token_ids, attention_masks, **kwargs)
            meta.update(m)
        else:
            bag_emb = self.no_mentions_emb

        bag_emb = self.bag_downprojection(bag_emb)


        gate1 = self.ff_gate(torch.cat([es1, bag_emb]))
        gate2 = self.ff_gate(torch.cat([es2, bag_emb]))
        gate3 = self.ff_gate(torch.cat([es3, bag_emb]))
        gate4 = self.ff_gate(torch.cat([es4, bag_emb]))

        meta.update({'gate1': gate1, 'gate2': gate2, 'gate3': gate3, 'gate4': gate4})

        h1 = gate1 * bag_emb + (1-gate1) * es1
        h2 = gate2 * bag_emb + (1-gate2) * es2
        h3 = gate3 * bag_emb + (1-gate3) * es2
        h4 = gate4 * bag_emb + (1-gate4) * es2

        scores = h1 @ r_re.t() + h2 @ r_re.t() + h3 @ r_im.t() - h4 @ r_im.t()

        return scores, meta

