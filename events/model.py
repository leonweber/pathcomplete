from collections import defaultdict
from operator import itemgetter
from torch.utils.data.dataloader import default_collate

from transformers.optimization import get_linear_schedule_with_warmup
from pathlib import Path
import logging

from flair.data import Sentence, Token
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, BertConfig
import networkx as nx
import numpy as np
from events.modeling_bert import TypedBertModel
from transformers import BertModel
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel

from events import consts
from events import dataset
from events.evaluation import Evaluator
from events.parse_standoff import StandoffAnnotation
from util.utils import overlaps


def get_event_trigger_positions(sent, ann):
    trigger_positions = []
    for trigger in ann.triggers.values():
        if sent.end_pos > int(trigger.start) >= sent.start_pos:
            trigger_positions.append(
                (
                    int(trigger.start) - sent.start_pos,
                    int(trigger.end) - sent.start_pos,
                )
            )

    return trigger_positions


def get_entity_trigger_positions(sent, ann):
    trigger_positions = []
    for trigger in ann.entity_triggers:
        if sent.end_pos > int(trigger.start) >= sent.start_pos:
            trigger_positions.append(
                (
                    int(trigger.start) - sent.start_pos,
                    int(trigger.end) - sent.start_pos,
                )
            )

    return trigger_positions


class TwoStageLoss(nn.Module):
    def forward(self, node_logits, edge_logits, labels):
        true_node_type = labels[0]
        seq_len = labels.size(0)
        log_likelihood_node_type = F.log_softmax(node_logits, dim=0)[true_node_type]
        p_edge_type = F.log_softmax(edge_logits[:, true_node_type], dim=1)
        log_likelihood_edge_type = p_edge_type[torch.arange(seq_len-1), labels[1:]]
        nll = -(log_likelihood_node_type + log_likelihood_edge_type.sum())

        return nll


class BertForGraphGeneration(nn.Module):
    def __init__(self, num_labels, num_node_types):
        super().__init__()
        self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.dropout = nn.Dropout(0.2)
        self.num_labels = num_labels
        self.num_node_types = num_node_types
        self.node_classifier = nn.Linear(768, self.num_node_types)
        self.edge_classifier = nn.Linear(768, self.num_labels*self.num_node_types)

    def forward(self, batch):
        if "token_type_ids" in batch:
            x = self.model.forward(
                input_ids=batch["input_ids"].long(),
                token_type_ids=batch["token_type_ids"].long(),
                attention_mask=batch["attention_mask"].long(),
            )[0]
        else:
            x = self.model.forward(
                input_ids=batch["input_ids"].long(),
                attention_mask=batch["attention_mask"].long(),
            )[0]

        x = self.dropout(x)
        node_logits = self.node_classifier(x[:, 0])
        bs, length = x.shape[:2]
        edge_logits = self.edge_classifier(x).reshape((bs, length) + (self.num_node_types, self.num_labels))

        return node_logits, edge_logits[:, 1:]


class EventExtractor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.i=0
        self.num_epochs = config["num_epochs"]

        self.train_path = config["train"]
        self.dev_path = config["dev"]
        self.lr = config["lr"]
        self.small = config["small"]

        DSType = getattr(dataset, config["dataset_type"])

        self.train_dataset = DSType(
            Path(config["train"]),
            config["bert"],
            small=self.small
        )

        if hasattr(self.train_dataset, "edge_type_to_id"):
            self.dev_dataset = DSType(
                Path(config["dev"]), config["bert"],
                small=self.small,
                edge_type_to_id=self.train_dataset.edge_type_to_id
            )
        else:
            self.dev_dataset = DSType(
                Path(config["dev"]), config["bert"],
                small=self.small
            )

        self.id_to_label_type = {
            v: k for k, v in self.train_dataset.label_to_id.items()
        }

        self.id_to_node_type = {
            v: k for k, v in self.train_dataset.node_type_to_id.items()
        }
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert"], use_fast=True)
        bert_config = BertConfig.from_pretrained(config["bert"])
        self.model = BertForGraphGeneration(num_labels=max(self.id_to_label_type) + 1, num_node_types=self.train_dataset.num_node_types)
        # self.tokenizer = BertTokenizerFast.from_pretrained(config["bert"])
        self.max_events_per_sentence = 10

    def forward(self, batch):
        return self.model(batch)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, batch_size=8, shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dev_dataset, batch_size=8, shuffle=False
        )
        return loader

    def train_eval_dataloader(self):
        self.train_dataset.predict = True
        loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x,
        )
        return loader

    def val_gold_dataloader(self):
        loader = DataLoader(
            self.dev_dataset, batch_size=32, shuffle=False, collate_fn=BioNLPDataset.collate_fn,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x,
        )
        return loader

    def training_step(self, batch, batch_idx):
        batch_node_logits, batch_edge_logits = self.forward(batch)

        full_loss = 0
        node_correct = []
        edge_correct = []
        correct_by_node_type = defaultdict(list) 
        loss_by_node_type = defaultdict(list) 
        id_to_node_type = {v: k for k, v in self.train_dataset.node_type_to_id.items()}
        for node_logits, edge_logits, input_ids, labels in zip(
                batch_node_logits,
                batch_edge_logits,
                batch["input_ids"],
                batch["labels"],
        ):
            pad_start = torch.where(input_ids == 0)[0][0]
            edge_logits = edge_logits[:pad_start-1]
            labels = labels[:pad_start]
            loss = TwoStageLoss()(node_logits, edge_logits, labels)
            full_loss += loss
            node_correct.append((node_logits.argmax() == labels[0]).item())
            edge_correct.append((edge_logits[:, labels[0]].argmax(dim=1) == labels[1:]).all().item())
            node_type = id_to_node_type[labels[0].item()]
            correct_by_node_type[node_type].append(node_correct[-1])
            loss_by_node_type[node_type].append(loss.item())

        full_loss /= len(batch["labels"])
        self.log("train_acc", np.mean(np.array(node_correct) & np.array(edge_correct)), prog_bar=True)
        self.log("train_node_acc", np.mean(node_correct), prog_bar=False)
        self.log("train_edge_acc", np.mean(edge_correct), prog_bar=False)
        self.log("train_loss", full_loss)

        for node_type in correct_by_node_type:
            self.log("train_node_acc_" + node_type, np.mean(correct_by_node_type[node_type]))
            self.log("train_loss_" + node_type, np.mean(loss_by_node_type[node_type]))

        return full_loss

    def validation_step(self, batch, batch_idx):
        batch_node_logits, batch_edge_logits = self.forward(batch)

        full_loss = 0
        node_correct = []
        edge_correct = []
        correct_by_node_type = defaultdict(list) 
        loss_by_node_type = defaultdict(list) 
        id_to_node_type = {v: k for k, v in self.train_dataset.node_type_to_id.items()}
        for node_logits, edge_logits, input_ids, labels in zip(
                batch_node_logits,
                batch_edge_logits,
                batch["input_ids"],
                batch["labels"],
        ):
            pad_start = torch.where(input_ids == 0)[0][0]
            edge_logits = edge_logits[:pad_start-1]
            labels = labels[:pad_start]
            loss = TwoStageLoss()(node_logits, edge_logits, labels)
            full_loss += loss
            node_correct.append((node_logits.argmax() == labels[0]).item())
            edge_correct.append((edge_logits[:, labels[0]].argmax(dim=1) == labels[1:]).all().item())
            node_type = id_to_node_type[labels[0].item()]
            correct_by_node_type[node_type].append(node_correct[-1])
            loss_by_node_type[node_type].append(loss.item())

        full_loss /= len(batch["labels"])
        self.log("val_acc", np.mean(np.array(node_correct) & np.array(edge_correct)), prog_bar=True)
        self.log("val_node_acc", np.mean(node_correct), prog_bar=False)
        self.log("val_edge_acc", np.mean(edge_correct), prog_bar=False)
        self.log("val_loss", full_loss)

        for node_type in correct_by_node_type:
            self.log("val_node_acc_" + node_type, np.mean(correct_by_node_type[node_type]))
            self.log("val_loss_" + node_type, np.mean(loss_by_node_type[node_type]))

        return full_loss

    def validation_step_gold(self, batch, batch_idx):
        batch_logits = self.forward(batch)
        eg_loss = 0
        result = defaultdict(int)
        for i, (logits, labels, input_ids) in enumerate(zip(batch_logits, batch["eg_labels"], batch["eg_input_ids"])):
            preds = logits.argmax(dim=1).cpu()
            labels = torch.tensor(labels).cpu()

            true_edges = (self.train_dataset.edge_type_to_id["None"] != labels).cpu()
            pred_edges = (self.train_dataset.edge_type_to_id["None"] != preds).cpu()

            if not any(true_edges) and not any(pred_edges):
                continue

            if any(true_edges) and any(pred_edges) and all(true_edges == pred_edges) and any(preds[true_edges].cpu() != labels[true_edges]):
                result["wrong_edge_type"] += 1

            if any(true_edges) and any(pred_edges) and any(true_edges != pred_edges):
                print()
                print("Wrong edge target: " + self.tokenizer.decode(input_ids.tolist()).replace("[PAD]", ""))
                print("\tShould have been: ")
                missed_edges = torch.tensor(batch["eg_node_spans_text"][i])[true_edges]
                missed_types = labels[true_edges]
                for edge, type in zip(missed_edges, missed_types):
                    start = edge[0]
                    end = edge[1]
                    print("\t\t" + self.id_to_edge_type[type.item()] + " " + self.tokenizer.decode(input_ids[start:end].tolist()) + ": " + self.id_to_node_type[batch["eg_node_types_text"][i][start].item()])
                print("\tWas: ")
                edges = torch.tensor(batch["eg_node_spans_text"][i])[pred_edges]
                types = preds[pred_edges]
                for edge, type in zip(edges, types):
                    start = edge[0]
                    end = edge[1]
                    print("\t\t" + self.id_to_edge_type[type.item()] + " " + self.tokenizer.decode(input_ids[start:end].tolist()) + ": " + self.id_to_node_type[batch["eg_node_types_text"][i][start].item()])
                result["wrong_edge_target"] += 1
                print()

            if any(true_edges) and not any(pred_edges):
                print()
                print("Stopped here: " + self.tokenizer.decode(input_ids.tolist()).replace("[PAD]", ""))
                print("\tShould have been: ")
                missed_edges = torch.tensor(batch["eg_node_spans_text"][i])[true_edges]
                missed_types = labels[true_edges]
                for edge, type in zip(missed_edges, missed_types):
                    start = edge[0]
                    end = edge[1]
                    print("\t\t" + self.id_to_edge_type[type.item()] + " " + self.tokenizer.decode(input_ids[start:end].tolist()) + ": " + self.id_to_node_type[batch["eg_node_types_text"][i][start].item()])

                result["stopped_too_early"] += 1
                print()

            if not any(true_edges) and any(pred_edges):
                result["predicted_too_many"] += 1


            if all(true_edges == pred_edges) and all(preds[true_edges].cpu() == labels[true_edges]):
                result["tp"] += 1
            elif not any(pred_edges):
                result["fn"] += 1
            elif not any(true_edges):
                result["fp"] += 1
            else:
                result["fp"] += 1
                result["fn"] += 1


            # if all(preds[pred_edges].cpu() != labels[pred_edges]):
            #     result["fp"] += 1
            #
            # if all(preds[true_edges].cpu() != labels[true_edges]):
            #     result["fn"] += 1

        return result

    def get_edges_from_labels(self, input_ids, labels, token_starts):
        edge_types = {}

        tags = [self.id_to_label_type[i.item()] for i in labels]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids,
                                                      skip_special_tokens=False)
        sentence = Sentence()
        for token, tag, start in zip(tokens[1:], tags, token_starts[1:]):
            token = Token(token.replace("##", ""), start_position=start)
            sentence.add_token(token)
            sentence.tokens[-1].add_label("Edge", tag)

        i_sep = [tok.idx for tok in sentence if tok.text == "[SEP]"][0]
        for span in sentence.get_spans("Edge"):
            if span[0].idx < i_sep:
                edge_types[(1, span.start_pos, span.end_pos)] = span.tag
            else:
                edge_types[(2, span.start_pos, span.end_pos)] = span.tag

        return edge_types

    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=0, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler
    
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        return (len(self.train_dataset) / 16) * self.num_epochs

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimizer = Adafactor(
        #     optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
        # )

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]


    def get_joint_prob_per_node_type(self, node_logits, edge_logits):
        node_probs = torch.log(torch.softmax(node_logits, dim=0))
        edge_probs_per_node_type = torch.softmax(edge_logits, dim=2)
        max_sequences = edge_probs_per_node_type.argmax(dim=2)
        edge_probs_per_node_type = edge_probs_per_node_type.reshape(-1, len(self.id_to_label_type))
        max_sequence_probs = edge_probs_per_node_type[torch.arange(len(edge_probs_per_node_type)), max_sequences.reshape(-1)]
        max_sequence_probs = max_sequence_probs.reshape_as(max_sequences)
        joint_sequence_probs = torch.sum(torch.log(max_sequence_probs), dim=0)

        joint_probs = node_probs[:len(joint_sequence_probs)] + joint_sequence_probs

        return joint_probs


    def predict(self, text, G_partial=None):
        beam_size = 5
        G_partial = G_partial or nx.DiGraph()
        G_beams = [(G_partial, 0.0)]
        with torch.no_grad():
            for i in range(self.max_events_per_sentence):
                G_beams_new = []
                G_beams_valid = []
                beam_examples = [] 
                for G_pred, prob_G in G_beams:
                    try:
                        example = self.train_dataset._build_example(node=None, G_full=G_pred, known_nodes=G_pred.nodes, text1=text)
                    except IndexError:
                        continue
                    beam_examples.append(example)
                    G_beams_valid.append((G_pred, prob_G))
                
                batch = default_collate(beam_examples)
                for k, v in batch.items():
                    try:
                        batch[k] = v.to(self.device)
                    except AttributeError:
                        pass

                pad_start = torch.where(batch["input_ids"] == 0)[1][0]
                node_logits, edge_logits = self.forward(batch)
                # pred_node_type_idx = node_logits[0].argmax().item()
                for node_logit, edge_logit, (G_pred, prob_G), example in zip(node_logits, edge_logits, G_beams_valid, beam_examples):
                    joint_probs = self.get_joint_prob_per_node_type(node_logit, edge_logit[:pad_start])
                    # joint_probs = torch.log_softmax(node_logit, dim=0)
                    for pred_node_type_idx, prob_step in enumerate(joint_probs):
                        G_pred_beam_step = G_pred.copy()
                        prob = prob_G + prob_step.item()
                        pred_node_type = self.id_to_node_type[pred_node_type_idx]
                        pred_edge_labels = edge_logit[:, pred_node_type_idx].argmax(dim=1)[:pad_start-1]
                        pred_edges = self.get_edges_from_labels(
                            input_ids=example["input_ids"][:pad_start],
                            labels=pred_edge_labels,
                            token_starts=example["offset_mapping"][:pad_start, 0].long().tolist()
                        )
                        if pred_node_type != "None" and pred_edges:
                            try:
                                self.train_dataset.add_node(G=G_pred_beam_step, node_type=pred_node_type, edges=pred_edges,
                                                            text=text)
                            except (ValueError, AssertionError):
                                continue
                        G_beams_new.append((G_pred_beam_step, prob))
                
                G_beams = sorted(G_beams_new, reverse=True, key=itemgetter(1))[:beam_size]

        return G_beams[0][0]




    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end( self, outputs ):
        return self.validation_epoch_end(outputs)

    def validation_epoch_end_gold(self, outputs):
        aggregated_outputs = defaultdict(int)
        for result in outputs:
            for k, v in result.items():
                aggregated_outputs[k] += v
        try:
            p = aggregated_outputs["tp"] / (aggregated_outputs["tp"] + aggregated_outputs["fp"])
            r = aggregated_outputs["tp"] / (aggregated_outputs["tp"] + aggregated_outputs["fn"])
            f1 = (2*p*r)/(p+r)
        except ZeroDivisionError:
            p = r = f1 = 0.0


        aggregated_outputs.update({
            "precision": p,
            "recall": r,
            "f1": f1,
            "log": {}
        })

        return dict(aggregated_outputs)


