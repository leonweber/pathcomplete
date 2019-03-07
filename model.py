import torch
from torch.nn import functional as F
from typing import Optional, Dict, List, Any

from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training import metrics
from overrides import overrides
from torch import nn


@Seq2VecEncoder.register("bertpooler")
class BertPooler(Seq2VecEncoder):

    def __init__(self, bert_dim):
        super(BertPooler, self).__init__()
        self.bert_dim =bert_dim


    def forward(self, embs: torch.tensor, mask: torch.tensor=None) -> torch.tensor:
        return embs[:, 0]

    @overrides
    def get_output_dim(self):
        return self.bert_dim


@Model.register('eventmatch')
class EventMatch(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(EventMatch, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.out_projection = nn.Linear(self.encoder.get_output_dim(), 2)

        self.metrics = {"accuracy": metrics.CategoricalAccuracy(), "f1": metrics.F1Measure(positive_label=1)}

        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self, text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None, metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        mask = util.get_text_field_mask(text)
        token_embeddings = self.text_field_embedder(text)
        text_embedding = self.encoder(token_embeddings, mask)

        logits = self.out_projection(text_embedding)
        output_dict = {"logits": logits, "probs": F.softmax(logits, dim=1)}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False):
        return {
            "f1": self.metrics["f1"].get_metric(reset=reset)[2],
            "accuracy": self.metrics["accuracy"].get_metric(reset=reset)
        }



