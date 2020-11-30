from collections import defaultdict
from pathlib import Path
import logging

from flair.data import Sentence, Token
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, BertConfig
import networkx as nx
import numpy as np

from events import consts
from events.dataset import BELDataset
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



class EventExtractor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.i=0

        self.train_path = config["train"]
        self.dev_path = config["dev"]
        self.linearize_events = config["linearize"]
        self.output_dir = config["output_dir"]
        self.use_dagger = config["use_dagger"]
        self.lr = config["lr"]

        self.small = config["small"]


        self.train_dataset = BELDataset(
            Path(config["train"]),
            config["bert"],
        )
        self.dev_dataset = BELDataset(
            Path(config["dev"]), config["bert"],
        )

        self.id_to_label_type = {
            v: k for k, v in self.train_dataset.label_to_id.items()
        }

        bert_config = BertConfig.from_pretrained(config["bert"])
        bert_config.num_labels = max(self.id_to_label_type) + 1
        self.bert = BertForTokenClassification.from_pretrained(config["bert"], config=bert_config)
        self.dropout = nn.Dropout(0.1)
        self.edge_classifier = nn.Linear(768, bert_config.num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert"])
        self.max_events_per_sentence = 30

    def forward(self, batch):

        logits = self.bert(
            input_ids=batch["input_ids"].long(),
            token_type_ids=batch["token_type_ids"].long(),
        )[0]

        return logits


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, batch_size=16, shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.dev_dataset, batch_size=16, shuffle=False
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
        batch_logits = self.forward(batch)

        loss = 0
        accs = []
        for logits, input_ids, labels in zip(
                batch_logits,
                batch["input_ids"],
                batch["labels"],
        ):
            pad_start = torch.where(input_ids == 0)[0][0]
            logits = logits[:pad_start]
            labels = labels[:pad_start]
            loss += nn.CrossEntropyLoss()(logits, labels)
            accs.append((logits.argmax(dim=1) == labels).all().long().item())
        loss /= batch_logits.size(0)
        self.log("train_acc", np.mean(accs), prog_bar=True)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_logits = self.forward(batch)

        loss = 0
        accs = []
        for logits, input_ids, labels in zip(
                batch_logits,
                batch["input_ids"],
                batch["labels"],
        ):
            pad_start = torch.where(input_ids == 0)[0][0]
            logits = logits[:pad_start]
            labels = labels[:pad_start]
            loss += nn.CrossEntropyLoss()(logits, labels)
            accs.append((logits.argmax(dim=1) == labels).all().long().item())
        loss /= batch_logits.size(0)
        self.log("val_acc", np.mean(accs), prog_bar=True)
        self.log("val_loss", loss)

        return loss

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
        for token, tag, start in zip(tokens, tags, token_starts):
            if token == "[CLS]":
                continue
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


    def predict(self, text):
        G_pred = nx.DiGraph()
        for i in range(self.max_events_per_sentence):
            example = self.train_dataset._build_example(node=None, G_full=G_pred, known_nodes=G_pred.nodes, text1=text)
            example["input_ids"] = example["input_ids"].unsqueeze(0).to(self.device)
            example["token_type_ids"] = example["token_type_ids"].unsqueeze(0).to(self.device)
            # example["attention_mask"] = example["attention_mask].squeez
            pad_start = torch.where(example["input_ids"] == 0)[1][0]
            logits = self.forward(example).squeeze(0)[:pad_start]
            pred_labels = logits.argmax(dim=1)
            node_type = self.id_to_label_type[pred_labels[0].item()]
            edges = self.get_edges_from_labels(
                input_ids=example["input_ids"].squeeze(0)[:pad_start],
                labels=pred_labels,
                token_starts=example["offset_mapping"][:pad_start, 0].long().tolist()
            )
            if node_type != "O" and edges:
                self.train_dataset.add_node(G=G_pred, node_type=node_type, edges=edges,
                                            text=text)

        return G_pred




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

