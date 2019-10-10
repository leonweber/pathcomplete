import transformers
from torch import nn
import torch

class DistantBert(nn.Module):
    def __init__(self, bert, n_classes):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(bert)
        self.ff_attention = nn.Linear(768, 1)
        self.ff_output = nn.Linear(768, n_classes)

    def forward(self, token_ids, attention_masks, entity_pos, labels, **kwargs):

        _, x = self.bert(token_ids, attention_masks)
        alphas = torch.sigmoid(self.ff_attention(x))
        x = torch.sum(alphas * x, dim=0)
        x = self.ff_output(x)

        return x
