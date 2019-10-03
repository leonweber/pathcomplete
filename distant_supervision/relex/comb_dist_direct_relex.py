from typing import Dict

import logging
from overrides import overrides
import torch
from torch import nn
import numpy as np
from allennlp.data import Vocabulary
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, CnnEncoder
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.f1_measure import F1Measure
from allennlp.modules import TextFieldEmbedder

from relex.multilabel_average_precision_metric import MultilabelAveragePrecision
from relex.relation_instances_reader import RelationInstancesReader
from relex.tensor_models import Simple, BagOnly

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("comb_dist_direct_relex")
class CombDistDirectRelex(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 cnn_size: int = 100,
                 tensor_model_size: int = 2000,
                 dropout_weight: float = 0.1,
                 with_entity_embeddings: bool = True,
                 sent_loss_weight: float = 1,
                 attention_weight_fn: str = 'sigmoid',
                 attention_aggregation_fn: str = 'max',
                 sent_encoder: Seq2VecEncoder = CnnEncoder,
                 embedding_size: int = 200) -> None:
        regularizer = None
        super().__init__(vocab, regularizer)
        self.num_classes = self.vocab.get_vocab_size("labels")

        self.text_field_embedder = text_field_embedder
        self.dropout_weight = dropout_weight
        self.with_entity_embeddings = with_entity_embeddings
        self.sent_loss_weight = sent_loss_weight
        self.attention_weight_fn = attention_weight_fn
        self.attention_aggregation_fn = attention_aggregation_fn
        self.sent_encoder = sent_encoder


        # instantiate position embedder
        pos_embed_output_size = 5
        pos_embed_input_size = 2 * RelationInstancesReader.max_distance + 1
        self.pos_embed = nn.Embedding(pos_embed_input_size, pos_embed_output_size)
        pos_embed_weights = np.array([range(pos_embed_input_size)] * pos_embed_output_size).T
        self.pos_embed.weight = nn.Parameter(torch.Tensor(pos_embed_weights))

        d = cnn_size
        cnn_output_size = d

        # dropout after word embedding
        self.dropout = nn.Dropout(p=self.dropout_weight)

        #  given a sentence, returns its unnormalized attention weight
        self.attention_ff = nn.Sequential(
            nn.Linear(cnn_output_size, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        self.ff_before_alpha = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

        bag_emb_size = cnn_output_size
        if self.with_entity_embeddings:
            bag_emb_size += embedding_size

        if tensor_model_size > 0:
            tensor_model_type = Simple
        else:
            tensor_model_type = BagOnly
        self.tensor_model = tensor_model_type(tensor_emb_size=tensor_model_size, bag_emb_size=bag_emb_size,
                                              n_entities=vocab.get_vocab_size('entities'),
                                              n_relations=self.vocab.get_vocab_size("labels"))

        self.bag_loss_fun = torch.nn.BCEWithLogitsLoss()  # sigmoid + binary cross entropy
        self.prov_loss_fun = torch.nn.BCEWithLogitsLoss()
        self.metrics = {}
        self.metrics['ap'] = MultilabelAveragePrecision()  # average precision = AUC
        self.metrics['bag_loss'] = Average()  # to display bag-level loss
        self.metrics['gate'] = Average()
        self.metrics['alpha'] = Average()
        # self.metrics['sent_ap'] = MultilabelAveragePrecision()
        # self.metrics['sent_ap'](torch.tensor([[1.0,0.0]]), torch.tensor([[0,1]]))
        #
        # if self.sent_loss_weight > 0:
        #     self.metrics['sent_loss'] = Average()  # to display sentence-level loss

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                entities: torch.LongTensor,
                mentions: Dict[str, torch.LongTensor],
                positions1: torch.LongTensor,
                positions2: torch.LongTensor,
                is_direct_supervision_bag: torch.LongTensor,
                has_mentions: torch.LongTensor,
                labels: torch.LongTensor,  # bag-level labels,
                pmids: torch.LongTensor,
                pmid_labels: torch.LongTensor,
                metadata=None,
                ) -> Dict[str, torch.Tensor]:

        positions1[positions1 < 0] = 0
        positions2[positions2 < 0] = 0

        no_mentions_mask = ~has_mentions.bool()

        alphas1, mask, x1 = self.bag_embedding(mentions=mentions, positions1=positions1, positions2=positions2,
                                               no_mentions_mask=no_mentions_mask)
        alphas2, mask, x2 = self.bag_embedding(mentions=mentions, positions1=positions2, positions2=positions1,
                                               no_mentions_mask=no_mentions_mask)

        logits, gate1, gate2 = self.tensor_model(entities['entities'], x1, x2, no_mentions_mask)

        bag_mask = mask.sum(dim=2) > 0
        unpadded_alphas1 = []
        for alpha, bag_m in zip(alphas1, bag_mask):
            if bag_m.sum() == 0:
                unpadded_alphas1.append(np.array([]))
            else:
                unpadded_alphas1.append(alpha.cpu().detach().numpy()[bag_m.cpu().numpy()])

        unpadded_alphas2 = []
        for alpha, bag_m in zip(alphas1, bag_mask):
            if bag_m.sum() == 0:
                unpadded_alphas2.append(np.array([]))
            else:
                unpadded_alphas2.append(alpha.cpu().detach().numpy()[bag_m.cpu().numpy()])

        output_dict = {'logits': logits, 'alphas1': unpadded_alphas1, 'alphas2': unpadded_alphas2,
                       'gate1': gate1,
                       'gate2': gate2}  # sigmoid is applied in the loss function and the metric class, not here

        if labels is not None:  # Training and evaluation
            alphas = (alphas1 * gate1.unsqueeze(-1) + alphas2 * gate2.unsqueeze(-1))/2
            all_pmid_scores = []
            all_pmid_labels = []
            for alphas_, pmids_, pmid_labels_ in zip(alphas, pmids, pmid_labels):
                pmid_scores = {}
                pmid_labels_map = {}
                for alpha, pmid, pmid_label in zip(alphas_, pmids_, pmid_labels_):
                    if pmid not in pmid_scores:
                        pmid_scores[pmid] = alpha
                    else:
                        pmid_scores[pmid] = pmid_scores[pmid] + alpha
                    pmid_labels_map[pmid] = pmid_label
                if pmids_:
                    all_pmid_scores.append(torch.cat([pmid_scores[pmid] for pmid in pmids_]))
                    all_pmid_labels.append(torch.stack([pmid_labels_map[pmid] for pmid in pmids_]))
            if all_pmid_scores:
                all_pmid_scores = torch.cat(all_pmid_scores).view(-1)
                all_pmid_labels = torch.cat(all_pmid_labels).view(-1)

            bag_loss = self.bag_loss_fun(logits, labels.squeeze(-1).type_as(
                logits)) * self.num_classes  # scale the loss to be more readable

            if hasattr(all_pmid_scores, 'nelement') and all_pmid_scores.nelement() > 0 and self.sent_loss_weight > 0:
                provenance_loss = self.prov_loss_fun(all_pmid_scores, all_pmid_labels.type_as(all_pmid_scores))
                loss = self.sent_loss_weight * provenance_loss + (1-self.sent_loss_weight)  * bag_loss
            else:
                loss = bag_loss

            self.metrics['bag_loss'](bag_loss.item())
            self.metrics['ap'](logits, labels.squeeze(-1))
            self.metrics['gate']((gate1 + gate2).mean().item()/2)
            self.metrics['alpha']((torch.sigmoid(alphas1) + torch.sigmoid(alphas2)).mean().item()/2)

            output_dict['loss'] = loss

        return output_dict

    def bag_embedding(self, mentions, positions1, positions2, no_mentions_mask):
        tokens = mentions['tokens']
        assert tokens.dim() == 3
        batch_size = tokens.size(0)
        padded_bag_size = tokens.size(1)
        padded_sent_size = tokens.size(2)
        mask = util.get_text_field_mask(mentions, num_wrapping_dims=1)
        # embed text
        t_embd = self.text_field_embedder(mentions)
        mask[no_mentions_mask] = mask[no_mentions_mask] * 0
        # embed position information
        p1_embd = self.pos_embed(positions1)
        p2_embd = self.pos_embed(positions2)
        # concatinate position emebddings to the word embeddings
        # x.shape: batch_size x padded_bag_size x padded_sent_size x (text_embedding_size + 2 * position_embedding_size)
        x = torch.cat([t_embd, p1_embd, p2_embd], dim=3)
        if self.dropout_weight > 0:
            x = self.dropout(x)
        # merge the first two dimensions becase sentence encoder doesn't support the 4d input
        x = x.view(batch_size * padded_bag_size, padded_sent_size, -1)
        mask = mask.view(batch_size * padded_bag_size, -1)
        # call sequence encoder
        x = self.sent_encoder(x, mask)  # (batch_size * padded_bag_size) x cnn_output_size
        # separate the first two dimensions back
        x = x.view(batch_size, padded_bag_size, -1)
        mask = mask.view(batch_size, padded_bag_size, -1)
        # compute unnormalized attention weights, one scaler per sentence
        unnorm_alphas = self.attention_ff(x)
        # apply a small FF to the attention weights
        alphas = self.ff_before_alpha(unnorm_alphas)
        # normalize attention weights based on the selected weighting function
        if self.attention_weight_fn == 'uniform':
            alphas = mask[:, :, 0].float()
        elif self.attention_weight_fn == 'softmax':
            alphas = util.masked_softmax(alphas.squeeze(-1), mask[:, :, 0].float())
        elif self.attention_weight_fn == 'sigmoid':
            alphas = torch.sigmoid(alphas.squeeze(-1)) * mask[:, :, 0].float()
        elif self.attention_weight_fn == 'norm_sigmoid':  # equation 7 in https://arxiv.org/pdf/1805.02214.pdf
            alphas = torch.sigmoid(alphas.squeeze(-1)) * mask[:, :, 0].float()
            alphas = alphas / alphas.sum(dim=-1, keepdim=True)
        else:
            assert False
        # Input:
        #   `x`: sentence encodings
        #   `alphas`: attention weights
        #   `attention_aggregation_fn`: aggregation function
        # Output: bag encoding
        if self.attention_aggregation_fn == 'max':
            x = alphas.unsqueeze(-1) * x  # weight sentences
            x = x.max(dim=1)[0]  # max pooling
        elif self.attention_aggregation_fn == 'avg':
            x = torch.bmm(alphas.unsqueeze(1), x).squeeze(1)  # average pooling
        else:
            assert False
        if self.with_entity_embeddings:
            # actual bag_size (not padded_bag_size) for each instance in the batch
            bag_size = mask[:, :, 0].sum(dim=1, keepdim=True).float()
            bag_size[bag_size == 0] = 1.0

            e1_mask = (positions1 == 0).long() * mask
            e1_embd = torch.matmul(e1_mask.unsqueeze(2).float(), t_embd)
            e1_embd_sent_sum = e1_embd.squeeze(dim=2).sum(dim=1)
            e1_embd_sent_avg = e1_embd_sent_sum / bag_size

            e2_mask = (positions2 == 0).long() * mask
            e2_embd = torch.matmul(e2_mask.unsqueeze(2).float(), t_embd)
            e2_embd_sent_sum = e2_embd.squeeze(dim=2).sum(dim=1)
            e2_embd_sent_avg = e2_embd_sent_sum / bag_size

            e1_e2_mult = e1_embd_sent_avg * e2_embd_sent_avg
            x = torch.cat([x, e1_e2_mult], dim=1)
        return unnorm_alphas, mask, x

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prob_thr = 0.0  # to ignore predicted labels with low prob.
        probs = torch.sigmoid(output_dict['logits'])
        output_dict['labels'] = []
        for row in probs.cpu().data.numpy():
            labels = []
            for i, p in enumerate(row):
                if p > prob_thr:  # ignore predictions with low prob.
                    labels.append((self.vocab.get_token_from_index(i, namespace="labels"), float(p)))
                    # output_dict[self.vocab.get_token_from_index(i, namespace="labels")] = torch.Tensor([float(p)])
            output_dict['labels'].append(labels)
        del output_dict['loss']
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "ap": self.metrics["ap"].get_metric(reset=reset),
            # "sent_ap": self.metrics["sent_ap"].get_metric(reset=reset),
            "bag_loss": self.metrics["bag_loss"].get_metric(reset=reset),
            # "sent_loss": self.metrics["sent_loss"].get_metric(reset=reset)
            "gate": self.metrics["gate"].get_metric(reset=reset),
            "alpha": self.metrics["alpha"].get_metric(reset=reset)
        }
