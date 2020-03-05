import torch
from torch import nn
from transformers import BertModel, PreTrainedModel, BertConfig
from transformers.modeling_bert import BertOnlyMLMHead, BertPreTrainedModel


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
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.mlm_head = BertOnlyMLMHead(config)

        self.init_weights()


    def forward(self, x):
        input_ids = x['input_ids']  # shape == (batch_size, sentences (3), length (512))
        attention_mask = x['attention_mask']
        batch_size, sentences, length = input_ids.shape
        e1_start = x['e1_start']  # shape == (batch_size, sentences (3))
        e2_start = x['e2_start']  # shape == (batch_size, sentences (3))
        embs = self.bert(input_ids=input_ids.view((-1, length)),
                         attention_mask=attention_mask.view((-1, length)))[
            0]  # shape == (batch_size * sentences, length (512), 768)

        mlm_logits = self.mlm_head(embs)

        e1_embs = embs[
            torch.arange(batch_size * sentences), e1_start.view(-1)]  # shape == (batch * sentences, 768)
        e2_embs = embs[
            torch.arange(batch_size * sentences), e2_start.view(-1)]  # shape == (batch * sentences, 768)
        embs = torch.cat([e1_embs, e2_embs], dim=1).reshape(
            (batch_size, sentences, -1))  # shape == (batch, sentences, 768*2)


        return embs[:, 0, ...], embs[:, 1, ...], embs[:, 2, ...], mlm_logits

