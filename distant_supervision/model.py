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
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.attention = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.parallel_bert = None

        self.init_weights()

    def forward(self, token_ids, attention_masks, entity_pos, **kwargs):
        outputs = self.parallel_bert(token_ids, attention_mask=attention_masks)
        x = outputs[1]

        unnorm_alphas = self.attention(x)

        alphas = torch.sigmoid(unnorm_alphas)
        x = torch.max(x * alphas, dim=0)[0]

        meta = {'alphas_hist': np.histogram(alphas.detach().cpu().numpy()), 'alphas': unnorm_alphas}

        x = self.classifier(x)

        return x, meta


