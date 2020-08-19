import os
from collections import defaultdict, deque
from glob import glob
import itertools
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from torch.utils.data import DataLoader
from torch_geometric.nn import GINEConv
import logging
from transformers import BertModel, BertTokenizerFast, BertConfig
import networkx as nx
from torch_geometric.data import Data
import torch_geometric
import numpy as np

from events import consts
from events.model_gnn import GNN
from events.dataset import (
    PC13Dataset,
    GE13Dataset,
    get_triggers,
    get_trigger_to_position,
    BioNLPDataset, get_free_event_id,
    get_a2_lines_from_graph,
    get_event_trigger_lines_from_sentences, get_trigger_spans, get_partial_graph
)
from events.evaluation import Evaluator
from events.modeling_bert import BertGNNModel
from events.parse_standoff import StandoffAnnotation

logging.basicConfig(level=logging.CRITICAL)

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
        self.batch_size = 64
        self.dev_path = config["dev"]
        self.linearize_events = config["linearize"]
        self.output_dir = config["output_dir"]
        self.loss_weight_td = config["loss_weight_td"] / (config["loss_weight_td"] + config["loss_weight_eg"])
        self.loss_weight_eg = config["loss_weight_eg"] / (config["loss_weight_td"] + config["loss_weight_eg"])
        self.max_span_width = 10
        self.bert = TransformerWordEmbeddings(config["bert"], fine_tune=True, layers="-1", batch_size=16)
        # if config["trigger_detector"].endswith(".pt"):
        #     self.trigger_detector = SequenceTagger.load(config["trigger_detector"])
        # elif os.path.isdir(config["trigger_detector"]):
        #     self.trigger_detector = {}
        #     for fname in Path(config["trigger_detector"]).glob("*a2"):
        #         with fname.open() as f:
        #             self.trigger_detector[fname.with_suffix(".txt").name] = [l.strip() for l in f if l.startswith("T")]

        DatasetType = get_dataset_type(config["train"])

        self.dropout = nn.Dropout(0.2)

        self.train_dataset = DatasetType(
            Path(config["train"]),
            config["bert"],
            linearize_events=self.linearize_events,
            trigger_ordering=config["trigger_ordering"],
            # trigger_detector = self.trigger_detector
        )
        self.dev_dataset = DatasetType(
            Path(config["dev"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_ordering=config["trigger_ordering"],
            # trigger_detector = self.trigger_detector
        )
        self.test_dataset = DatasetType(
            Path(config["test"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_ordering=config["trigger_ordering"],
            # trigger_detector = self.trigger_detector
        )

        bert_config = BertConfig.from_pretrained(config["bert"])
        bert_config.node_type_vocab_size = len(self.train_dataset.node_type_to_id)
        bert_config.edge_type_vocab_size = len(self.train_dataset.edge_type_to_id)
        # self.bert = BertGNNModel.from_pretrained(config["bert"], config=bert_config)
        # self.bert = BertModel.from_pretrained(config["bert"], config=bert_config)
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

        self.node_dim = 2*self.bert.embedding_length + 100
        self.edge_dim = 100
        self.node_type_embedding = nn.Embedding(len(self.id_to_node_type), 100)
        self.edge_type_embedding = nn.Embedding(len(self.id_to_edge_type), 100)
        self.edge_classifier = nn.Linear(2*self.node_dim, len(self.train_dataset.edge_type_to_id))
        self.graph_embedder = GNN(num_layer=3, node_emb_dim=self.node_dim, gnn_type="gin",
                                  edge_emb_dim=self.edge_dim)


    def split_args(self, graph):
        # TODO check whether this all-combinations strategy yields too many results and leads to bad scores

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
            if not edge_types:
                graph.remove_node(event)
            elif "Theme" not in edge_types and event_type not in self.train_dataset.NO_THEME_ALLOWED:
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

    def clean_up_graph(self, graph: nx.DiGraph):
        old_nodes = None
        while old_nodes != list(graph.nodes()):
            lift_event_edges(graph)
            self.remove_invalid_edges(graph)
            break_up_cycles(graph)
            self.split_args(graph)
            self.remove_invalid_events(graph)
            old_nodes = list(graph.nodes())


    def get_graphs(self, batch):
        graph_batch = []
        nx_graphs = []
        for sentence, trigger, graph, event_id in zip(
                batch["eg_sentence"],
                batch["eg_trigger"],
                batch["eg_graph"],
                batch["eg_event_id"]
        ):
            graph: nx.DiGraph
            graph = graph.copy()
            graph.add_node(event_id, type=graph.nodes[trigger]["type"], span=graph.nodes[trigger]["span"])
            trigger_embs = []
            node_types = []
            for node, d in graph.nodes(data=True):
                trigger_embs.append(
                    torch.cat([sentence.tokens[d["span"][0]].embedding,
                               sentence.tokens[d["span"][1]].embedding])
                )
                node_types.append(self.train_dataset.node_type_to_id[d["type"]])
            node_type_embs = self.node_type_embedding(torch.tensor(node_types).long().to(self.device))
            node_embs = torch.cat([torch.stack(trigger_embs), node_type_embs], dim=1)
            for node in graph.nodes:
                graph.add_edge(node, node, type="Self")
                if node != event_id:
                    graph.add_edge(event_id, node, type="Candidate")

            edge_types = []
            for u, v, data in graph.edges(data=True):
                edge_types.append(self.train_dataset.edge_type_to_id[data["type"]])

            # TODO maybe also add span embeddings for src and trgt here
            edge_embs = self.edge_type_embedding(torch.tensor(edge_types).long().to(self.device))
            graph_data = torch_geometric.utils.from_networkx(graph)
            graph_data.x = node_embs
            graph_data.edge_attr = edge_embs

            nx_graphs.append(graph)

            graph_batch.append(graph_data)

        return torch_geometric.data.Batch.from_data_list(graph_batch).to(self.device), nx_graphs



    def forward(self, batch):
        graphs, nx_graphs = self.get_graphs(batch)
        node_embs = self.graph_embedder(graphs.x, graphs.edge_index, graphs.edge_attr)
        edge_embs = torch.cat([node_embs[graphs.edge_index][0], node_embs[graphs.edge_index][1]], dim=1)
        edge_logits = self.edge_classifier(edge_embs)

        return edge_logits, nx_graphs


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
        self.bert.embed(batch["eg_sentence"])
        batch_logits, nx_graphs = self.forward(batch)
        edge_labels = []
        edge_candidate_mask = []
        for graph, gold_edges in zip(nx_graphs, batch["eg_edges_to_predict"]):
            for u, v, d in graph.edges(data=True):
                if d["type"] == "Candidate":
                    edge_label = gold_edges.get((u, v), "None")
                    edge_labels.append(self.train_dataset.edge_type_to_id[edge_label])
                    edge_candidate_mask.append(True)
                else:
                    edge_candidate_mask.append(False)
        edge_labels = torch.tensor(edge_labels).long().to(self.device)
        edge_candidate_mask = torch.tensor(edge_candidate_mask).bool().to(self.device)
        eg_loss = nn.CrossEntropyLoss()(batch_logits[edge_candidate_mask], edge_labels)
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
            self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=BioNLPDataset.collate_fn
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
        self.bert.embed(sentences)
        a2_trigger_lines = [l.strip() for l in ann.a2_lines if l.startswith("T")]
        # a2_trigger_lines = self.predict_triggers(sentences)

        # if isinstance(self.trigger_detector, dict):
        #     a2_trigger_lines = self.trigger_detector[fname]
        # else:
        #     a2_trigger_lines = get_event_trigger_lines_from_sentences(sentences, self.trigger_detector, len(a1_lines))
        if self.loss_weight_eg == 0: # this model was only trained for trigger detection
            return "\n".join(a2_trigger_lines)
        if len(a2_trigger_lines) > 100: # this should only happen before any training started
            a2_trigger_lines = []
        a2_event_lines = []
        ann = StandoffAnnotation(a1_lines, a2_trigger_lines)
        predicted_graph = ann.event_graph.copy()
        for sentence in sentences:
            entity_triggers, event_triggers = get_triggers(sentence, ann)
            trigger_spans = get_trigger_spans(entity_triggers + event_triggers, sentence)
            events_in_sentence = set()

            for i_trigger, marked_trigger in enumerate(self.train_dataset.trigger_ordering(event_triggers, ann)):
                for i_generated in range(self.max_events_per_trigger):
                    ann = StandoffAnnotation(a1_lines, a2_trigger_lines + a2_event_lines)

                    event_id = get_free_event_id(predicted_graph)
                    batch = {"eg_sentence": [sentence],
                             "eg_trigger": [marked_trigger.id],
                             "eg_graph": [get_partial_graph(ann, events_in_sentence,
                                                            triggers=entity_triggers + event_triggers,
                                                            trigger_spans = trigger_spans)],
                             "eg_event_id": [event_id]}


                    edge_logits, graphs = self(batch)
                    edge_types = {}
                    for logits, (u, v, d) in zip(edge_logits, graphs[0].edges(data=True)):
                        if d["type"] == "Candidate":
                            edge_types[(u,v)] = self.id_to_edge_type[logits.argmax(dim=0).item()]

                    if all(i == "None" for i in edge_types.values()):
                        break
                    else:
                        predicted_graph.add_node(event_id, type=marked_trigger.type)
                        predicted_graph.add_edge(marked_trigger.id, event_id, type="Trigger")
                        for (u, v), edge_type in edge_types.items():
                            if edge_type != "None":
                                predicted_graph.add_edge(u, v, type=edge_type)

                        self.clean_up_graph(predicted_graph)
                        a2_event_lines = get_a2_lines_from_graph(predicted_graph)

        return "\n".join(a2_trigger_lines + a2_event_lines)


    def predict_batched(self, text, ann, fname):
        a1_lines = ann.a1_lines
        sentences = self.train_dataset.sentence_splitter.split(text)
        self.bert.embed(sentences)
        a2_trigger_lines = [l.strip() for l in ann.a2_lines if l.startswith("T")]
        # a2_trigger_lines = self.predict_triggers(sentences)

        # if isinstance(self.trigger_detector, dict):
        #     a2_trigger_lines = self.trigger_detector[fname]
        # else:
        #     a2_trigger_lines = get_event_trigger_lines_from_sentences(sentences, self.trigger_detector, len(a1_lines))
        if self.loss_weight_eg == 0: # this model was only trained for trigger detection
            return "\n".join(a2_trigger_lines)
        if len(a2_trigger_lines) > 100: # this should only happen before any training started
            a2_trigger_lines = []
        a2_event_lines = []
        ann = StandoffAnnotation(a1_lines, a2_trigger_lines)
        predicted_graph = ann.event_graph.copy()
        sentence_to_constant_batch = []

        sentence_to_trigger_queue = []
        sentence_to_events_in_sentence = []
        for sentence in sentences:
            sentence_to_trigger_queue.append(deque())
            sentence_to_events_in_sentence.append(set())
            entity_triggers, event_triggers = get_triggers(sentence, ann)

            for trigger in self.train_dataset.trigger_ordering(event_triggers, ann):
                sentence_to_trigger_queue[-1].append(trigger)

        sentence_to_trigger_count = [0] * len(sentences)
        batch = defaultdict(list)
        while any(i for i in sentence_to_trigger_queue) > 0: # while there is still some unprocessed trigger
            current_triggers = []
            ann = StandoffAnnotation(a1_lines, a2_trigger_lines + a2_event_lines)
            for i_sentence, sentence in enumerate(sentences):
                entity_triggers, event_triggers = get_triggers(sentence, ann)
                trigger_spans = get_trigger_spans(entity_triggers + event_triggers, sentence)

                trigger_queue = sentence_to_trigger_queue[i_sentence]

                if sentence_to_trigger_count[i_sentence] >= self.max_events_per_trigger and trigger_queue:
                    trigger_queue.popleft()
                    sentence_to_trigger_count[i_sentence] = 0

                if not trigger_queue:
                    continue

                marked_trigger = trigger_queue[0]
                sentence_to_trigger_count[i_sentence] += 1

                event_id = get_free_event_id(predicted_graph)

                batch["eg_sentence"].append(sentence)
                batch["eg_trigger"].append(marked_trigger.id)
                batch["eg_graph"].append(get_partial_graph(ann, sentence_to_events_in_sentence[i_sentence],
                                                      triggers=entity_triggers + event_triggers,
                                                      trigger_spans = trigger_spans))
                batch["eg_event_id"].append(event_id)

            edge_logits, graphs = self(batch)
            edge_types = {}

            for logitss, graph, event_id, marked_trigger in zip(edge_logits, graphs, batch["eg_event_id"], batch["eg_trigger"]):
                for logits, (u, v, d) in zip(logitss, graph.edges(data=True)):
                    if d["type"] == "Candidate":
                        edge_types[(u,v)] = self.id_to_edge_type[logits.argmax(dim=0).item()]

                if all(i == "None" for i in edge_types.values()):
                    break
                else:
                    predicted_graph.add_node(event_id, type=graph.nodes[marked_trigger]["type"])
                    predicted_graph.add_edge(marked_trigger, event_id, type="Trigger")
                    for (u, v), edge_type in edge_types.items():
                        if edge_type != "None":
                            predicted_graph.add_edge(u, v, type=edge_type)

                    self.clean_up_graph(predicted_graph)
                    a2_event_lines = get_a2_lines_from_graph(predicted_graph)

        return "\n".join(a2_trigger_lines + a2_event_lines)
