import itertools
import math
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from flair.models import SequenceTagger
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast, BertConfig
import networkx as nx
import numpy as np

from events import consts
from events.dataset import (
    PC13Dataset,
    GE13Dataset,
    get_text_encoding_and_node_spans,
    get_triggers,
    get_trigger_to_position,
    get_event_linearization,
    get_event_graph,
    get_adjacency_matrix, MAX_LEN, BioNLPDataset, get_free_event_id,
    get_a2_lines_from_graph,
    get_event_trigger_lines_from_sentences
)
from events.evaluation import Evaluator
from events.modeling_bert import BertGNNModel
from events.parse_standoff import StandoffAnnotation

BERTS = {"bert": BertModel, "gnn": BertGNNModel}

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

def lift_event_edges(graph):
    for trigger in [n for n in graph.nodes if n.startswith("T")]:
        events = [v for _, v, d in graph.out_edges(trigger, data=True) if d["type"] == "Trigger"]
        if events:
            for u, _, d in list(graph.in_edges(trigger, data=True)):
                for event in events:
                    graph.add_edge(u, event, **d)
                graph.remove_edge(u, trigger)


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

        self.train_path = config["train"]
        self.dev_path = config["dev"]
        self.linearize_events = config["linearize"]
        self.output_dir = config["output_dir"]
        self.loss_weight_td = config["loss_weight_td"] / (config["loss_weight_td"] + config["loss_weight_eg"])
        self.loss_weight_eg = config["loss_weight_eg"] / (config["loss_weight_td"] + config["loss_weight_eg"])
        self.max_span_width = 10
        if config["trigger_detector"].endswith(".pt"):
            self.trigger_detector = SequenceTagger.load(config["trigger_detector"])
        elif os.path.isdir(config["trigger_detector"]):
            self.trigger_detector = {}
            for fname in Path(config["trigger_detector"]).glob("*a2"):
                with fname.open() as f:
                    self.trigger_detector[fname.with_suffix(".txt").name] = [l.strip() for l in f if l.startswith("T")]

        DatasetType = get_dataset_type(config["train"])

        self.dropout = nn.Dropout(0.2)

        self.train_dataset = DatasetType(
            Path(config["train"]),
            config["bert"],
            linearize_events=self.linearize_events,
            trigger_ordering=config["trigger_ordering"],
            trigger_detector = self.trigger_detector,
            small=config["small"]
        )
        self.dev_dataset = DatasetType(
            Path(config["dev"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_ordering=config["trigger_ordering"],
            trigger_detector = self.trigger_detector,
            small=config["small"]
        )
        self.test_dataset = DatasetType(
            Path(config["test"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_ordering=config["trigger_ordering"],
            trigger_detector = self.trigger_detector,
            small=config["small"]
        )

        bert_config = BertConfig.from_pretrained(config["bert"])
        bert_config.node_type_vocab_size = len(self.train_dataset.node_type_to_id)
        bert_config.edge_type_vocab_size = len(self.train_dataset.edge_type_to_id)
        # self.bert = BertGNNModel.from_pretrained(config["bert"], config=bert_config)
        self.bert = BertModel.from_pretrained(config["bert"], config=bert_config)
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert"])

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["@"]})
        # self.trigger_classifier = nn.Linear(768*3, len(self.train_dataset.node_type_to_id))
        self.loss_fn = nn.CrossEntropyLoss()
        self.span_poolings = nn.ModuleList([nn.AvgPool1d(kernel_size=i, stride=1) for i in range(1, self.max_span_width+1)])
        self.max_events_per_trigger = 10

        self.id_to_edge_type = {
            v: k for k, v in self.train_dataset.edge_type_to_id.items()
        }
        self.id_to_node_type = {
            v: k for k, v in self.train_dataset.node_type_to_id.items()
        }

        self.node_type_embedding = nn.Embedding(len(self.id_to_node_type), 100)
        self.hidden1 = nn.Linear(768*3 + self.node_type_embedding.embedding_dim*2,
                                         1000)
        self.hidden2 = nn.Linear(1000, 1000)
        self.edge_classifier = nn.Linear(1000, len(self.train_dataset.edge_type_to_id))

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

    def remove_invalid_events(self, graph):
        for event in [n for n in graph.nodes if n.startswith("E")]:
            edge_types = [i[2]["type"] for i in graph.out_edges(event, data=True)]
            event_type = graph.nodes[event]["type"]
            triggers = [u for u, v, d in graph.in_edges(event, data=True) if d["type"] == "Trigger"]
            if not edge_types:
                graph.remove_node(event)
            elif "Theme" not in edge_types and event_type not in self.train_dataset.NO_THEME_ALLOWED:
                graph.remove_node(event)
            elif not triggers:
                graph.remove_node(event)

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
                events = [v for _, v, d in graph.out_edges(trigger, data=True) if d["type"] == "Trigger"]
                for u, _, d in list(graph.in_edges(trigger, data=True)):
                    for event in events:
                        graph.add_edge(u, event, **d)
                    if events or remove_unlifted:
                        graph.remove_edge(u, trigger)


    def clean_up_graph(self, graph: nx.DiGraph, remove_unlifted=False,
                   remove_invalid=False):
        old_nodes = None
        while old_nodes != list(graph.nodes()):
            old_nodes = list(graph.nodes())
            self.lift_event_edges(graph, remove_unlifted)

            if remove_invalid:
                self.remove_invalid_edges(graph)
            break_up_cycles(graph)
            self.split_args(graph)

            if remove_invalid:
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
        attention_mask = self.get_full_attention_mask(batch["eg_input_ids"]).long()
        node_type_ids = []
        for node_types_text, node_types_graph in zip(batch["eg_node_types_text"], batch["eg_node_types_graph"]):
            node_type_ids.append(torch.cat([node_types_text, node_types_graph]))
        node_type_ids = torch.stack(node_type_ids).long()

        xs, _ = self.bert(
            input_ids=batch["eg_input_ids"].long(),
            # edge_type_ids=edge_type_ids,
            attention_mask=attention_mask,
            token_type_ids=batch["eg_token_type_ids"].long(),
            # node_type_ids=node_type_ids,
            # position_ids=position_ids
        )
        xs = self.dropout(xs)
        batch_logits = []
        for x, node_spans_text, node_types_text, trigger_span in zip(
            xs,
            batch["eg_node_spans_text"],
            batch["eg_node_types_text"],
            batch["eg_trigger_span"],
        ):
            x_nodes = []
            for node_span in node_spans_text:
                type_emb = self.node_type_embedding(node_types_text[node_span[0]].long())
                trigger_type_emb = self.node_type_embedding(node_types_text[trigger_span[0]].long())
                span_rep = [x[node_span[0]],
                            torch.mean(x[node_span[0] : node_span[1]], dim=0),
                            x[node_span[1]]-1,
                            type_emb,
                            trigger_type_emb]
                x_nodes.append(torch.cat(span_rep))
            x_nodes = torch.stack(x_nodes)
            x_nodes = self.hidden1(x_nodes)
            x_nodes = self.hidden2(x_nodes)
            logits = self.edge_classifier(x_nodes)
            batch_logits.append(logits)
        return batch_logits

    def forward_triggers(
            self,
            batch
    ):

        embeddings = self.bert(
            batch["td_input_ids"], attention_mask=batch["td_attention_mask"],
        )[0]
        span_embeddings = []
        span_starts = []
        span_ends = []
        embeddings_t = embeddings.transpose(1,2)
        for pool in self.span_poolings:
            span_width = pool.kernel_size[0]
            pooled_embeddings = pool(embeddings_t).transpose(1,2)
            start_embeddings = embeddings[:, :pooled_embeddings.size(1)]
            end_embeddings = embeddings[:, span_width-1:]
            span_embedding = torch.cat([start_embeddings, pooled_embeddings, end_embeddings], dim=2)
            span_embeddings.append(span_embedding)
            span_starts.append(torch.arange(0, pooled_embeddings.size(1)).repeat([pooled_embeddings.size(0), 1]))
            span_ends.append(torch.arange(span_width, embeddings.size(1)+1).repeat([pooled_embeddings.size(0), 1]))
        span_embeddings = torch.cat(span_embeddings, dim=1)
        span_starts = torch.cat(span_starts, dim=1)
        span_ends = torch.cat(span_ends, dim=1)

        return {"logits": self.trigger_classifier(span_embeddings),
                "span_starts": span_starts,
                "span_ends": span_ends}

    def training_step(self, batch, batch_idx):
        batch_logits = self.forward(batch)
        eg_loss = 0
        for i, (logits, labels) in enumerate(zip(batch_logits, batch["eg_labels"])):
            labels = torch.tensor(labels).to(self.device)
            eg_loss += self.loss_fn(logits, labels)
        eg_loss /= len(batch)

        # td_out = self.forward_triggers(batch)
        # logits = td_out["logits"].reshape(-1, td_out["logits"].size(-1))
        # labels = batch["td_labels"].reshape(-1, batch["td_labels"].size(-1)).float()
        # mask = batch["td_span_mask"].reshape(-1).bool()
        # loss_fn = nn.BCEWithLogitsLoss()
        # td_loss = loss_fn(logits[mask], labels[mask])

        # check whether input -> label mapping is correct
        # for batch_idx, span_idx, label_idx in zip(*torch.where(batch["td_labels"])):
        #     start = td_out["span_starts"][batch_idx, span_idx]
        #     end = td_out["span_ends"][batch_idx, span_idx]
        #     token = self.tokenizer.decode(batch["td_input_ids"][batch_idx, start:end].tolist())
        #     print(f"{token}: {self.id_to_node_type[label_idx.item()]}")

        # loss = self.loss_weight_td * td_loss + self.loss_weight_eg * eg_loss
        loss = eg_loss

        #
        # log = {"train_loss_eg": eg_loss, "train_loss_td": td_loss, "train_loss": loss,
        #        "max_td_logit": logits.max()}
        log = {"train_loss": loss}

        return {"loss": loss, "log": log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, batch_size=16, shuffle=True, collate_fn=BioNLPDataset.collate_fn
        )
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
        log.update(evaluator.evaluate_event_generation(aggregated_outputs))
        log.update(evaluator.evaluate_trigger_detection(aggregated_outputs))
        print(log)

        return {
            "val_f1": torch.tensor(log["f1"]),
            "val_f1_td": torch.tensor(log["f1_td"]),
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

    def predict_triggers(self, sentences):
        sentence_texts = [s.to_original_text() for s in sentences]
        encoding = self.tokenizer.batch_encode_plus(sentence_texts, padding=True,return_offsets_mapping=True)
        batch = {k: torch.tensor(v, device=self.device) for k, v in encoding.items()}
        renamed_batch = {}
        for k, v in batch.items():
            renamed_batch["td_" + k] = v
        out = self.forward_triggers(renamed_batch)

        predicted_lines = []
        trigger_classes = out["logits"] > 0.0
        predicted_indices = torch.where(trigger_classes)
        predicted_starts = out["span_starts"][predicted_indices[:2]]
        predicted_ends = out["span_starts"][predicted_indices[:2]]
        predicted_starts_char = batch["offset_mapping"][[predicted_indices[0], predicted_starts]][:, 0]
        predicted_ends_char = batch["offset_mapping"][[predicted_indices[0], predicted_ends]][:, 1]
        for i, (i_sent, start, end, cls) in enumerate(zip(predicted_indices[0], predicted_starts_char, predicted_ends_char, predicted_indices[2])):
            start = start.item()
            end = end.item()
            if start + end > 0:
                text = sentence_texts[i_sent][start:end]
                start += sentences[i_sent].start_pos
                end += sentences[i_sent].start_pos
                span_type = self.id_to_node_type[cls.item()]
                if span_type != "None":
                    predicted_lines.append(f"T{i}\t{span_type} {start} {end}\t{text}")

        return predicted_lines



    def predict(self, text, ann, fname):
        a1_lines = ann.a1_lines
        sentences = self.train_dataset.sentence_splitter.split(text)
        # a2_trigger_lines = [l.strip() for l in ann.a2_lines if l.startswith("T")]
        # a2_trigger_lines = self.predict_triggers(sentences)

        if isinstance(self.trigger_detector, dict):
            a2_trigger_lines = self.trigger_detector[fname]
        else:
            a2_trigger_lines = get_event_trigger_lines_from_sentences(sentences, self.trigger_detector, len(a1_lines))
        if self.loss_weight_eg == 0: # this model was only trained for trigger detection
            return "\n".join(a2_trigger_lines)
        if len(a2_trigger_lines) > 100: # this should only happen before any training started
            a2_trigger_lines = []
        a2_event_lines = []
        ann = StandoffAnnotation(a1_lines, a2_trigger_lines)
        predicted_graph = ann.event_graph.copy()
        for sentence in sentences:
            entity_triggers, event_triggers = get_triggers(sentence, ann)
            trigger_to_position = get_trigger_to_position(sentence, ann)
            events_in_sentence = set()

            for i_trigger, marked_trigger in enumerate(self.train_dataset.trigger_ordering(event_triggers, ann)):
                for i_generated in range(self.max_events_per_trigger):
                    ann = StandoffAnnotation(a1_lines, a2_trigger_lines + a2_event_lines)
                    event_ids_in_sentence = []

                    event_to_trigger = {}
                    for event in ann.events.values():
                        event_to_trigger[event.id] = event.trigger.id
                        if event.trigger.id in event_triggers:
                            event_ids_in_sentence.append(event.id)

                    if self.linearize_events:
                        (
                            encoding_graph,
                            node_types_graph,
                            node_spans_graph,
                            node_ids_graph,
                        ) = get_event_linearization(ann=ann,
                                                    tokenizer=self.train_dataset.tokenizer,
                                                    node_type_to_id=self.train_dataset.node_type_to_id,
                                                    known_events=event_ids_in_sentence,
                                                    edge_types_to_mod=self.train_dataset.EDGE_TYPES_TO_MOD)
                    else:
                        (
                            encoding_graph,
                            node_types_graph,
                            node_spans_graph,
                            node_ids_graph,
                        ) = get_event_graph(ann=ann,
                                            tokenizer=self.train_dataset.tokenizer,
                                            entities=entity_triggers,
                                            node_type_to_id=self.train_dataset.node_type_to_id,
                                            known_events=event_ids_in_sentence)

                    remaining_length = MAX_LEN - len(encoding_graph["input_ids"])
                    if remaining_length <= 0:
                        print("Graph is too large for MAX_LENGTH. Skipping...")
                        continue
                    node_spans_graph = [(a + remaining_length, b + remaining_length) for a,b in node_spans_graph] # because we will append them to the text encoding
                    encoding_text, node_spans_text, node_types_text = get_text_encoding_and_node_spans(
                        text=sentence.to_original_text(),
                        trigger_pos=trigger_to_position[marked_trigger],
                        tokenizer=self.tokenizer,
                        max_length=remaining_length,
                        nodes=entity_triggers + event_triggers,
                        trigger_to_position=trigger_to_position,
                        ann=ann,
                        node_type_to_id=self.train_dataset.node_type_to_id
                    )
                    adjacency_matrix = get_adjacency_matrix(event_graph=ann.event_graph, nodes_text=entity_triggers + event_triggers,
                                                            nodes_graph=node_ids_graph, event_to_trigger=event_to_trigger,
                                                            edge_type_to_id=self.train_dataset.edge_type_to_id)
                    assert adjacency_matrix["text_to_graph"].shape[1] == len(node_spans_graph)
                    current_batch = {}
                    current_batch["eg_input_ids"] = torch.cat([torch.tensor(encoding_text["input_ids"]),
                                           torch.tensor(encoding_graph["input_ids"])]).unsqueeze(0).to(self.device)
                    current_batch["eg_token_type_ids"] = torch.zeros(current_batch["eg_input_ids"].size(0)).unsqueeze(0).to(self.device)
                    current_batch["eg_token_type_ids"][:, len(encoding_text["input_ids"]):] = 1

                    current_batch["eg_node_spans_text"] = torch.tensor([node_spans_text]).to(self.device)
                    current_batch["eg_node_spans_graph"] = torch.tensor([node_spans_graph]).to(self.device)
                    current_batch["eg_node_types_text"] = node_types_text.unsqueeze(0).to(self.device)
                    current_batch["eg_node_types_graph"] = node_types_graph.unsqueeze(0).to(self.device)
                    current_batch["eg_trigger_span"] = torch.tensor(node_spans_text[(entity_triggers + event_triggers).index(marked_trigger)]).unsqueeze(0).to(self.device)

                    current_batch["eg_adjacency_matrix"] = [adjacency_matrix]


                    edge_logits = self(current_batch)
                    edge_types = [
                        self.id_to_edge_type[i.item()]
                        for i in edge_logits[0].argmax(dim=1)
                    ]

                    if all(i == "None" for i in edge_types):
                        break
                    else:
                        event_id = get_free_event_id(predicted_graph)
                        predicted_graph.add_node(event_id, type=ann.triggers[marked_trigger].type)
                        predicted_graph.add_edge(marked_trigger, event_id, type="Trigger")
                        for edge_type, edge_trigger in zip(
                            edge_types, entity_triggers + event_triggers
                        ):
                            if edge_type != "None":
                                predicted_graph.add_edge(event_id, edge_trigger, type=edge_type)

                        self.clean_up_graph(predicted_graph, remove_unlifted=False, remove_invalid=False)
                        a2_event_lines = get_a2_lines_from_graph(predicted_graph)

        self.clean_up_graph(predicted_graph, remove_unlifted=False, remove_invalid=False)
        a2_event_lines = get_a2_lines_from_graph(predicted_graph)

        return "\n".join(a2_trigger_lines + a2_event_lines)
