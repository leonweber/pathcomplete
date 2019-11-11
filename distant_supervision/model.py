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


class BertForDistantSupervision(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(BertForDistantSupervision, self).__init__(config)
        self.bert = transformers.BertModel(config)
        self.parallel_bert = nn.DataParallel(self.bert)
        self.key = nn.Linear(768, 64)
        self.value = nn.Linear(768, 64)
        self.query = nn.Embedding(1, 64)
        self.classifier = nn.Linear(64, config.num_labels)
        self.sentinel_embedding = nn.Embedding(1, 768)

        self.init_weights()

    def forward(self, token_ids, attention_masks, entity_pos, **kwargs):
        outputs = self.parallel_bert(token_ids, attention_mask=attention_masks)
        x = outputs[1]
        sentinel = self.sentinel_embedding(torch.tensor(0).to(x.device))
        x = torch.cat([x, sentinel.unsqueeze(0)])

        key = self.key(x)
        value = self.value(x)
        query = self.query(torch.tensor(0).to(x.device))

        unnorm_alphas = (key @ query.t()) / math.sqrt(64)

        alphas = torch.softmax(unnorm_alphas, dim=0)
        x = (value.t() @ alphas.unsqueeze(1)).squeeze(1)

        meta = {'alphas_hist': np.histogram(alphas.detach().cpu().numpy()), 'alphas': unnorm_alphas}

        x = self.classifier(x)

        return x, meta


