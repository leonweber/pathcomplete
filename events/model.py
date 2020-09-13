import itertools
import math
import os
from collections import defaultdict
from pprint import pprint
from glob import glob
from pathlib import Path
import logging

from flair.data import Sentence, Token
import pytorch_lightning as pl
import torch
import torch.nn as nn
from flair.models import SequenceTagger
from pytorch_lightning.trainer.trainer import _PatchDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizerFast, BertConfig
import networkx as nx
import numpy as np

from events import consts
from events.dataset import (
    PC13Dataset,
    GE13Dataset,
    get_text_encoding_and_node_spans,
    get_trigger_to_position,
    get_event_linearization,
    get_event_graph,
    get_adjacency_matrix, MAX_LEN, BioNLPDataset, get_free_event_id,
    get_a2_lines_from_graph, get_free_trigger_id, filter_graph_to_sentence,
)
from events.evaluation import Evaluator
from events.modeling_bert import BertGNNModel
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


def break_up_cycles(graph):
    """
    Break up cycles by removing the first non-Theme edge that is found
    If it's an all-Theme cycle, just remove some edge
    """
    while True:
        try:
            last_edge = None
            for edge in nx.find_cycle(graph):
                last_edge = edge
                if graph.edges[edge]["type"] != "Theme":
                    graph.remove_edge(*edge)
                    break
            else:
                graph.remove_edge(*last_edge)
        except nx.NetworkXNoCycle:
            return


def get_dataset_type(dataset_path: str) -> BioNLPDataset:
    if "2013_GE" in dataset_path:
        # return GE13Dataset
        return PC13Dataset
    elif "2013_PC" in dataset_path:
        return PC13Dataset





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

        DatasetType = get_dataset_type(config["train"])


        self.train_dataset = DatasetType(
            Path(config["train"]),
            config["bert"],
            linearize_events=self.linearize_events,
            event_order=config["event_order"],
            small=config["small"]
        )
        self.dev_dataset = DatasetType(
            Path(config["dev"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            event_order=config["event_order"],
            small=config["small"]
        )
        self.test_dataset = DatasetType(
            Path(config["test"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            event_order=config["event_order"],
            small=config["small"]
        )

        self.id_to_label_type = {
            v: k for k, v in self.train_dataset.label_to_id.items()
        }

        bert_config = BertConfig.from_pretrained(config["bert"])
        bert_config.num_labels = max(self.id_to_label_type) + 1
        bert_config.node_type_vocab_size = len(self.train_dataset.node_type_to_id)
        self.bert = BertGNNModel.from_pretrained(config["bert"], config=bert_config)
        # self.bert = BertForTokenClassification.from_pretrained(config["bert"], config=bert_config)
        self.dropout = nn.Dropout(0.1)
        self.token_classifier = nn.Linear(768, bert_config.num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert"])

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["@"]})
        self.max_events_per_trigger = 10


    def split_args(self, graph):
        for event in [n for n in graph.nodes if n.startswith("E")]:
            singular_edges_by_type = defaultdict(set)
            valid_edges = []
            event_type = graph.nodes[event]["type"]
            for u, v, edge_data in graph.out_edges(event, data=True):
                edge_type = edge_data["type"]
                edge = u, v, edge_type
                if (event_type, edge_type) not in self.train_dataset.DUPLICATES_ALLOWED:
                    singular_edges_by_type[edge_type].add(edge)
                else:
                    valid_edges.append(edge)

            multiple_args_by_type = {}
            for type, edges in singular_edges_by_type.items():
                if len(edges) == 1:
                    valid_edges.append(list(edges)[0])
                else:
                    multiple_args_by_type[type] = edges

            if multiple_args_by_type.values():
                products = list(itertools.product(*multiple_args_by_type.values()))
                for product in products:
                    new_event = get_free_event_id(graph)
                    graph.add_node(new_event, type=event_type)

                    # add out-edges of new event
                    for old_event, v, edge_type in itertools.chain(product, valid_edges):
                        graph.add_edge(new_event, v, type=edge_type)

                    # add in-edges of old event
                    for u, old_event, edge_data in graph.in_edges(event, data=True):
                        edge_type = edge_data["type"]
                        graph.add_edge(u, new_event, type=edge_type)

                # delete old event if we had to split
                graph.remove_node(event)

    def remove_invalid_nodes(self, graph):
        for node in [n for n in graph.nodes]:
            if graph.nodes[node]["type"] == "None":
                graph.remove_node(node)


    def remove_invalid_events(self, graph):
        for event in [n for n in graph.nodes if n.startswith("E")]:
            edge_types = [i[2]["type"] for i in graph.out_edges(event, data=True)]
            event_type = graph.nodes[event]["type"]

            triggers = [v for _, v, d in graph.out_edges(event, data=True) if d["type"] == "Trigger"]
            if not edge_types:
                graph.remove_node(event)
                continue

            elif "Theme" not in edge_types and event_type not in self.train_dataset.NO_THEME_ALLOWED:
                graph.remove_node(event)
                continue

            elif not triggers:
                graph.remove_node(event)
                continue

    def remove_invalid_edges(self, graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))
        for event in [n for n in graph.nodes if n.startswith("E")]:
            event_type = graph.nodes[event]["type"]
            for u, v, d in list(graph.out_edges(event, data=True)):
                edge_type = d["type"]
                v_type = graph.nodes[v]["type"]
                if not self.train_dataset.is_valid_argument_type(event_type=event_type, arg=edge_type, reftype=v_type, refid=v):
                    graph.remove_edge(u, v)

    def lift_event_edges(self, graph, remove_unlifted):
        for trigger in [n for n in graph.nodes if n.startswith("T")]:
            if graph.nodes[trigger]["type"] in self.train_dataset.EVENT_TYPES:
                events = [u for u, _, d in graph.in_edges(trigger, data=True) if d["type"] == "Trigger"]
                for u, _, d in list(graph.in_edges(trigger, data=True)):
                    if d["type"] != "Trigger":
                        for event in events:
                            graph.add_edge(u, event, **d)
                        if events or remove_unlifted:
                            graph.remove_edge(u, trigger)


    def clean_up_graph(self, graph: nx.DiGraph, remove_unlifted=False,
                   remove_invalid=False, lift=False):
        old_nodes = None
        while old_nodes != list(graph.nodes()):
            old_nodes = list(graph.nodes())

            if lift:
                self.lift_event_edges(graph, remove_unlifted)

            if remove_invalid:
                self.remove_invalid_edges(graph)
            break_up_cycles(graph)
            self.split_args(graph)

            if remove_invalid:
                self.remove_invalid_nodes(graph)
                self.remove_invalid_events(graph)

    def adjacency_matrix_to_edge_types(self, adjacency_matrix,
                                       node_spans_text,
                                       node_spans_graph,
                                       input_ids,
                                       sep_id
                                       ):
        batch_size, length = input_ids.shape
        attention_mask = torch.zeros((batch_size, length, length))
        attention_mask[:] = self.train_dataset.edge_type_to_id["None"]

        for i, input_id in enumerate(input_ids):
            x = torch.where(input_id == sep_id)[0][0]
            attention_mask[i, :x+1, :x+1] = self.train_dataset.edge_type_to_id["InText"] # Allow intra-text attention

        for i_batch, (m, spans_text, spans_graph) in enumerate(zip(adjacency_matrix,
                                                                 node_spans_text,
                                                                 node_spans_graph)):
            text_to_graph = m["text_to_graph"]
            graph_to_graph = m["graph_to_graph"]

            for i_u, edge_types in enumerate(text_to_graph):
                start_u, end_u = spans_text[i_u]
                for i_v, edge_type in enumerate(edge_types):
                    start_v, end_v = spans_graph[i_v]
                    attention_mask[i_batch, start_u:end_u, start_v:end_v] = edge_type

            for i_u, edge_types in enumerate(graph_to_graph):
                start_u, end_u = spans_graph[i_u]
                for i_v, edge_type in enumerate(edge_types):
                    start_v, end_v = spans_graph[i_v]
                    attention_mask[i_batch, start_u:end_u, start_v:end_v] = edge_type

        return attention_mask

    def get_full_attention_mask(self, input_ids):
        attention_mask = torch.ones_like(input_ids, device=self.device).long()
        attention_mask[torch.where(input_ids == 0)] = 0

        return attention_mask

    def get_gnn_position_ids(self, input_ids, sep_id):
        input_shape = input_ids.shape
        position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        for i, input_id in enumerate(input_ids):
            x = torch.where(input_id == sep_id)[0][0]
            position_ids[i, x+1:] = 0 # FIXME Maybe 0 isn't that good here, because it's shared with CLS?

        return position_ids

    def forward(self, batch):
        attention_mask = self.get_full_attention_mask(batch["input_ids"]).long()

        xs = self.bert(
            input_ids=batch["input_ids"].long(),
            attention_mask=attention_mask,
            token_type_ids=batch["token_type_ids"].long(),
            node_type_ids=batch["node_type_ids"].long()
        )[0]

        # return xs

        xs = self.dropout(xs)

        token_logits = self.token_classifier(xs)

        return token_logits


    def training_step(self, batch, batch_idx):
        batch_logits = self.forward(batch)

        loss = 0
        for logits, input_ids, labels in zip(
                batch_logits,
                batch["input_ids"],
                batch["labels"],
        ):
            seq1_end = torch.where(input_ids == self.tokenizer.sep_token_id)[0][0]
            loss += nn.CrossEntropyLoss()(logits[:seq1_end], labels[:seq1_end])

        loss /= len(batch)

        log = {"train_loss": loss}

        return {"loss": loss, "log": log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, batch_size=16, shuffle=True, collate_fn=BioNLPDataset.collate_fn
        )
        # loader = DataLoader(
        #     self.train_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x
        # )
        return loader

    def validation_step(self, batch, batch_idx):
        fname, text, ann = batch[0]
        return {fname: self.predict(text, ann, fname)}

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

    def validation_epoch_end(self, outputs):
        aggregated_outputs = {}
        for i in outputs:
            aggregated_outputs.update(i)

        evaluator = Evaluator(
            eval_script=consts.PC13_EVAL_SCRIPT,
            data_dir=self.dev_path,
            out_dir=self.output_dir/"eval",
            result_re=consts.PC13_RESULT_RE,
            verbose=True,
        )
        log = {}
        for k, v in evaluator.evaluate_event_generation(aggregated_outputs).items():
            log["val_" + k] = v
        for k, v in evaluator.evaluate_trigger_detection(aggregated_outputs).items():
            log["val_" + k] = v

        print(log)

        return {
            "val_f1": torch.tensor(log["val_f1"]),
            "val_f1_td": torch.tensor(log["val_f1_td"]),
            "log": log
        }

    def val_dataloader(self):
        loader = DataLoader(
            self.dev_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x,
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

    def find_overlap(self, current_span, trigger_spans):
        for i, trigger_span in enumerate(trigger_spans):
            if max(0, min(current_span[1], trigger_span[1]) - max(current_span[0], trigger_span[0])) > 0:
                return i
        return None

    def get_edge_types_from_logits(self, input_ids, logits, trigger_spans, trigger_ids, token_starts):
        edge_types = {}

        seq1_end = torch.where(input_ids[0] == self.tokenizer.sep_token_id)[0][0]
        tags = [self.id_to_label_type[i.item()] for i in logits[0].argmax(dim=1)[:seq1_end]]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][:seq1_end],
                                                      skip_special_tokens=False)
        sentence = Sentence()
        for token, tag, start in zip(tokens, tags, token_starts):
            token = Token(token.replace("##", ""), start_position=start)
            sentence.add_token(token)
            sentence.tokens[-1].add_label("Edge", tag)
        logging.debug(sentence.to_tagged_string("Edge"))

        for span in sentence.get_spans("Edge"):
            edge_types[(span.start_pos, span.end_pos)] = span.tag

        return edge_types

    def predict(self, text, ann, fname, return_batches=False):
        a1_lines = ann.a1_lines
        a2_lines = []
        sentences = self.train_dataset.sentence_splitter.split(text)
        batches = []
        ann = StandoffAnnotation(a1_lines, a2_lines)
        predicted_graph = ann.event_graph.copy()
        logging.debug("Predicting " + fname)
        for sentence in sentences:
            for i_generated in range(self.max_events_per_trigger):
                # ann = StandoffAnnotation(a1_lines, a2_lines)
                graph = filter_graph_to_sentence(predicted_graph, sentence)
                entity_triggers, event_triggers = self.train_dataset.get_triggers(sentence, graph)

                (
                    encoding_graph,
                    node_types_graph,
                    node_spans_graph,
                    node_ids_graph,
                ) = get_event_linearization(graph=graph,
                                            tokenizer=self.train_dataset.tokenizer,
                                            node_type_to_id=self.train_dataset.node_type_to_id,
                                            edge_types_to_mod=self.train_dataset.EDGE_TYPES_TO_MOD,
                                            event_ordering=self.train_dataset.event_ordering)
                remaining_length = MAX_LEN - len(encoding_graph["input_ids"])
                if remaining_length <= 0:
                    print("Graph is too large for MAX_LENGTH. Skipping...")
                    continue

                trigger_to_position = get_trigger_to_position(sentence, graph)
                encoding_text, node_spans_text, node_types_text, token_starts = get_text_encoding_and_node_spans(
                    text=sentence.to_original_text(),
                    trigger_pos=None,
                    tokenizer=self.tokenizer,
                    max_length=remaining_length,
                    graph=graph,
                    trigger_to_position=trigger_to_position,
                    node_type_to_id=self.train_dataset.node_type_to_id,
                    return_token_starts=True
                )

                token_starts = np.array(token_starts)
                token_starts += sentence.start_pos

                current_batch = {}
                current_batch["input_ids"] = torch.cat([torch.tensor(encoding_text["input_ids"]),
                                       torch.tensor(encoding_graph["input_ids"])]).unsqueeze(0).to(self.device)
                current_batch["token_type_ids"] = torch.zeros_like(current_batch["input_ids"]).to(self.device)
                current_batch["token_type_ids"][:, len(encoding_text["input_ids"]):] = 1
                current_batch["node_type_ids"] = torch.cat([node_types_text, node_types_graph]).unsqueeze(0).to(self.device)
                current_batch["ann"] = [ann]
                current_batch["sentence"] = [sentence]
                current_batch["encoding_graph"] = [encoding_graph]
                current_batch["fname"] = [fname]

                foo = []
                id_to_node_type = {v: k for k, v in self.train_dataset.node_type_to_id.items()}
                for tok, nt in zip(self.tokenizer.convert_ids_to_tokens(current_batch["input_ids"][0].tolist()), current_batch["node_type_ids"][0]):
                    foo.append((tok, id_to_node_type[nt.item()]))

                if return_batches:
                    batches.append(current_batch)

                logging.debug(self.tokenizer.decode(current_batch["input_ids"][0].tolist(), skip_special_tokens=True))

                logits = self(current_batch).cpu()

                edge_types = self.get_edge_types_from_logits(logits=logits,
                                                             input_ids=current_batch["input_ids"],
                                                             trigger_ids=entity_triggers + event_triggers,
                                                             trigger_spans=node_spans_text,
                                                             token_starts=token_starts)
                pretty_edge_types = {}
                for k, v in edge_types.items():
                    pretty_edge_types[text[k[0]:k[1]]] = v
                logging.debug(pretty_edge_types)

                if not edge_types:
                    break
                else:
                    triggers = []
                    for span, edge_type in edge_types.items():
                        if edge_type in self.train_dataset.EVENT_TYPES:
                            triggers.append((span, edge_type))

                    if len(triggers) != 1:
                        break # This is really bad, but what to do here?

                    event_id = get_free_event_id(predicted_graph)

                    trigger = triggers[0]

                    trigger_id = self.get_trigger(span=trigger[0],
                                                  graph=predicted_graph,
                                                  text=text,
                                                  type=trigger[1])
                    predicted_graph.add_node(event_id, type=predicted_graph.nodes[trigger_id]["type"])
                    predicted_graph.add_edge(event_id, trigger_id, type="Trigger")

                    for dst, edge_type in edge_types.items():
                        if edge_type in self.train_dataset.EDGE_TYPES:
                            dst_trigger = self.get_trigger(span=dst, type=None, graph=predicted_graph, text=text)
                            if dst_trigger != trigger_id: # otherwise it would overwrite the trigger edge
                                predicted_graph.add_edge(event_id, dst_trigger, type=edge_type)

                    self.clean_up_graph(predicted_graph, remove_unlifted=False, remove_invalid=False, lift=False)
                    # a2_lines = get_a2_lines_from_graph(predicted_graph, self.train_dataset.EVENT_TYPES)
                    # logging.debug("\n".join(a2_lines))
                    # logging.debug("done.")

        self.clean_up_graph(predicted_graph, remove_unlifted=True, remove_invalid=True, lift=True)
        a2_lines = get_a2_lines_from_graph(predicted_graph, self.train_dataset.EVENT_TYPES)

        if return_batches:
            return "\n".join(a2_lines), batches
        else:
            return "\n".join(a2_lines)

    def training_epoch_end( self, outputs ):
        self.i += 1

        if self.use_dagger or self.i % 5 == 0:
            self.eval()
            with torch.no_grad():
                # for example in self.train_dataset.examples:
                #     foo = {}
                #     foo["input_ids"] = example["input_ids"].unsqueeze(0).to(self.device)
                #     foo["token_type_ids"] = example["token_type_ids"].unsqueeze(0).to(self.device)
                #     seq1_end = torch.where(example["input_ids"] == self.tokenizer.sep_token_id)[0][0]
                #     labels_pred = self.forward(foo)[0].argmax(dim=1).cpu()
                #     if (labels_pred[:seq1_end] != example["labels"][:seq1_end]).any():
                #         foo["input_ids"] = example["input_ids"]
                #         foo["labels"] = labels_pred
                #         self.train_dataset.print_example(example)
                #         self.train_dataset.print_example(foo)
                #         print()
                predictions = {}
                all_batches = []
                for fname, text, ann in tqdm(self.train_dataset.predict_example_by_fname.values(),
                                             desc="Evaluating on train"):
                    predictions[fname], batches = self.predict(text, ann, fname, return_batches=True)
                    all_batches += batches

                self.train()
                evaluator = Evaluator(
                    eval_script=consts.PC13_EVAL_SCRIPT,
                    data_dir=self.train_path,
                    out_dir=self.output_dir/"eval",
                    result_re=consts.PC13_RESULT_RE,
                    verbose=True,
                )
                log = {}
                for k, v in evaluator.evaluate_event_generation(predictions).items():
                    log["train_" + k] = v
                for k, v in evaluator.evaluate_trigger_detection(predictions).items():
                    log["train_" + k] = v

                print(log)

                if self.use_dagger:
                    self.train_dataset.add_dagger_examples(all_batches)

                return {
                    "train_f1": torch.tensor(log["train_f1"]),
                    "train_f1_td": torch.tensor(log["train_f1_td"]),
                    "log": log}
        else:
            return {}

    def get_trigger(self, span, type, graph, text):
        span = (str(span[0]), str(span[1]))
        for n, d in graph.nodes(data=True):
            if n.startswith("T") and overlaps(d["span"], span):
                if d["type"] == "None": # type was unknown until now, set it
                    d["type"] = str(type)

                if d["type"] == type or type is None:
                    return n
        else:
            trigger_id = get_free_trigger_id(graph)
            graph.add_node(trigger_id, type=str(type), span=span,
                                     text=text[int(span[0]):int(span[1])])

            return trigger_id

