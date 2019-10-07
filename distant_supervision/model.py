import transformers
from torch import nn
import torch

class DistantBert(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(bert)
        self.ff_attention = nn.Linear(768, 1)

    def forward(self, sample):
        token_ids = sample['token_ids'].squeeze(0)[:10].to(self.device)
        attention_mask = sample['attention_masks'].squeeze(0)[:10].to(self.device)
        entity_pos = sample['entity_pos'].squeeze(0).to(self.device)

        _, x = self.bert(token_ids, attention_mask)
        alphas = torch.sigmoid(self.ff_attention(x))
        x = torch.sum(alphas * x, dim=1)

        return x
