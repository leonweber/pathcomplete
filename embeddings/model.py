import torch
from torch import nn
from transformers import BertModel, PreTrainedModel, BertConfig


class BertEntity(PreTrainedModel):
    def __init__(self, bert, n_classes):
        config = BertConfig.from_pretrained(bert)
        super().__init__(config)
        self.bert = BertModel.from_pretrained(bert)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768*2, n_classes)
        self.n_classes = n_classes

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



class SentenceMatcher(nn.Module):
    def __init__(self, dataset, bert_path):
        super().__init__()
        self.data = dataset

        self.bert = BertModel.from_pretrained(bert_path)
        self.output_layer = nn.Linear(768 * 3 * 2, 1)

    def forward(self, x):
        input_ids = x['input_ids']  # shape == (batch_size, sentences (3), length (512))
        attention_mask = x['attention_mask']
        batch_size, sentences, length = input_ids.shape
        e1_start = x['e1_start']  # shape == (batch_size, sentences (3))
        e2_start = x['e2_start']  # shape == (batch_size, sentences (3))
        embs = self.bert(input_ids=input_ids.view((-1, length)),
                         attention_mask=attention_mask.view((-1, length)))[

            0]  # shape == (batch_size * sentences, length (512), 768)
        e1_embs = embs[
            torch.arange(batch_size * sentences), e1_start.view(-1)]  # shape == (batch * sentences, 768)
        e2_embs = embs[
            torch.arange(batch_size * sentences), e2_start.view(-1)]  # shape == (batch * sentences, 768)
        embs = torch.cat([e1_embs, e2_embs], dim=1).reshape(
            (batch_size, sentences, -1))  # shape == (batch, sentences, 768*2)

        return embs[:, 0, ...], embs[:, 1, ...], embs[:, 2, ...]

    def training_step(self, batch, batch_idx):
        anchor, pos, neg = self(batch)
        loss = self.loss_fun(anchor, pos, neg)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

