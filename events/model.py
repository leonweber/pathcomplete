import os
from collections import defaultdict, deque
import math
from copy import deepcopy
from glob import glob
import itertools
from pathlib import Path

import dgl
import pytorch_lightning as pl
import torch
import torch.nn as nn
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerWordEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from torch.utils.data import DataLoader
import logging
from transformers import BertModel, BertTokenizerFast, BertConfig
import networkx as nx
import numpy as np

from events import consts
from events.model_gnn import GNN, MLPReadout, MDNReadout
from events.dataset import (
    PC13Dataset,
    GE13Dataset,
    get_triggers,
    get_trigger_to_position,
    BioNLPDataset, get_free_event_id,
    get_a2_lines_from_graph,
    get_event_trigger_lines_from_sentences, get_trigger_spans, get_partial_graph,
    get_free_trigger_id
)
from events.evaluation import Evaluator
from events.modeling_bert import BertGNNModel
from events.parse_standoff import StandoffAnnotation

logging.basicConfig(level=logging.CRITICAL)

BERTS = {"bert": BertModel, "gnn": BertGNNModel}



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
        self.batch_size = 32
        self.dev_path = config["dev"]
        self.linearize_events = config["linearize"]
        self.output_dir = config["output_dir"]

        self.loss_weight_td = config["loss_weight_td"]
        self.loss_weight_tt = config["loss_weight_tt"]
        self.loss_weight_eg = config["loss_weight_eg"]
        weight_sum = self.loss_weight_td + self.loss_weight_tt + self.loss_weight_eg
        self.loss_weight_td = self.loss_weight_td / weight_sum
        self.loss_weight_tt = self.loss_weight_tt / weight_sum
        self.loss_weight_eg = self.loss_weight_eg / weight_sum

        self.max_span_width = 10
        # self.embeddings = TransformerWordEmbeddings(config["bert"], fine_tune=True, layers="-1", batch_size=16)
        self.embeddings =  WordEmbeddings("pubmed")

        tag_dictionary = Dictionary(add_unk=False)
        tag_dictionary.multi_label = False
        tag_dictionary.add_item("O")
        tag_dictionary.add_item("B-Trigger")
        tag_dictionary.add_item("I-Trigger")
        self.node_detector = SequenceTagger(hidden_size=None,
                                            embeddings=self.embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type="trigger",
                                            use_crf=False,
                                            use_rnn=False)

        DatasetType = get_dataset_type(config["train"])

        self.dropout = nn.Dropout(config["dropout"])

        self.train_dataset = DatasetType(
            Path(config["train"]),
            config["bert"],
            linearize_events=self.linearize_events,
            trigger_ordering=config["trigger_ordering"],
            small=config["small"]
            # trigger_detector = self.trigger_detector
        )
        self.dev_dataset = DatasetType(
            Path(config["dev"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_ordering=config["trigger_ordering"],
            small=config["small"]
            # trigger_detector = self.trigger_detector
        )
        self.test_dataset = DatasetType(
            Path(config["test"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_ordering=config["trigger_ordering"],
            small=config["small"]
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

        self.hidden_dim = config["hidden_dim"]
        self.node_type_embedding = nn.Embedding(len(self.id_to_node_type), self.hidden_dim)
        self.edge_type_embedding = nn.Embedding(len(self.id_to_edge_type)*2, self.hidden_dim) # forward and reverse edges
        self.node_embedder = nn.Sequential(
            nn.Linear(2*self.embeddings.embedding_length, self.hidden_dim),
            nn.Dropout(config["dropout"]),
            nn.GELU()
        )
        self.graph_embedder = GNN(node_dim=self.hidden_dim, edge_dim=self.hidden_dim,
                                  dropout=config["dropout"], n_layers=3, layer_type=config["gnn"])
        self.edge_classifier = MDNReadout(self.hidden_dim, len(self.id_to_edge_type), n_components=20)

        self.node_classifier = nn.Linear(self.hidden_dim, len(self.id_to_node_type))

        # self.edge_classifier = nn.Linear(self.node_dim*2,
        #                                  len(self.id_to_edge_type))

    def mdn_loss(self, dgl_graphs, nx_graphs, all_gold_edges, eps=1e-8):
        nlls = []
        for dgl_graph, nx_graph, gold_edges in zip(dgl_graphs, nx_graphs, all_gold_edges):
            edge_labels = []
            candidate_mask = []
            for u, v, d in nx_graph.edges(data=True):
                if d["type"] == "Candidate":
                    edge_label = gold_edges.get((u, v), "None")
                    edge_labels.append(self.train_dataset.edge_type_to_id[edge_label])
                    candidate_mask.append(True)
                else:
                    candidate_mask.append(False)
            true_edge_probs_per_component = dgl_graph.edata["probs"][candidate_mask][torch.arange(sum(candidate_mask)), :, edge_labels]

            coeffs = torch.softmax(dgl_graph.edata["mixture_coefficients"].sum(dim=0), dim=0)
            joint_prob = sum( coeffs * true_edge_probs_per_component.prod(dim=0))
            nlls.append(-torch.log(joint_prob + eps))

        return sum(nlls) / len(nlls)

    def nll_loss(self, dgl_graphs, nx_graphs, all_gold_edges, eps=1e-8):
        nlls = []
        for dgl_graph, nx_graph, gold_edges in zip(dgl_graphs, nx_graphs, all_gold_edges):
            edge_labels = []
            candidate_mask = []
            for u, v, d in nx_graph.edges(data=True):
                if d["type"] == "Candidate":
                    edge_label = gold_edges.get((u, v), "None")
                    edge_labels.append(self.train_dataset.edge_type_to_id[edge_label])
                    candidate_mask.append(True)
                else:
                    candidate_mask.append(False)
            edge_labels = torch.tensor(edge_labels).long().to(self.device)
            logits = dgl_graph.edata["logits"][candidate_mask][:, 0, :].squeeze(1)

            nlls.append(nn.CrossEntropyLoss(reduction="sum")(logits, edge_labels))

        return sum(nlls) / len(nlls)






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


    def get_graphs(self, batch):
        nx_graphs = []
        dgl_graphs = []
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
            node_embs = self.node_embedder(torch.stack(trigger_embs))
            node_types = torch.tensor(node_types).to(self.device)

            graph.add_edge(trigger, event_id, type="Trigger")
            for node in graph.nodes:
                graph.add_edge(node, node, type="Self")
                if node.startswith("T"):
                    graph.add_edge(event_id, node, type="Candidate")

            edge_types = []
            for u, v, data in graph.edges(data=True):
                edge_types.append(self.train_dataset.edge_type_to_id[data["type"]])

            for u, v, d in graph.edges(data=True):
                if (v, u) not in graph.edges:
                    graph.add_edge(v, u, type=d["type"] + "_r")
                    edge_types.append(self.train_dataset.edge_type_to_id[d["type"]] + len(self.train_dataset.edge_type_to_id))

            edge_embs = self.edge_type_embedding(torch.tensor(edge_types).long().to(self.device))

            us = []
            vs = []
            for u, v in nx.convert_node_labels_to_integers(graph).edges:
                us.append(u)
                vs.append(v)
            dgl_graph = dgl.graph((us, vs)).to(self.device)
            dgl_graph.ndata["h"] = node_embs
            dgl_graph.ndata["type"] = node_types
            dgl_graph.edata["e"] = edge_embs

            dgl_graphs.append(dgl_graph)
            nx_graphs.append(graph)

        return dgl.batch(dgl_graphs), nx_graphs



    def forward(self, batch):
        batched_graph, nx_graphs = self.get_graphs(batch)
        self.graph_embedder(batched_graph, batched_graph.ndata["h"], batched_graph.edata["e"])
        graphs = self.edge_classifier(dgl.unbatch(batched_graph))
        # batched_graph.ndata["logits"] = self.node_classifier(batched_graph.ndata["h"])

        return nx_graphs, graphs

    def training_step(self, batch, batch_idx):
        # self.bert.embed(batch["eg_sentence"])
        # td_loss = self.node_detector.forward_loss(batch["eg_sentence"]) # this also embeds the sentence
        self.embeddings.embed(batch["eg_sentence"])
        nx_graphs, dgl_graphs = self.forward(batch)
        edge_labels = []
        edge_candidate_mask = []
        edge_labels = torch.tensor(edge_labels).long().to(self.device)
        edge_candidate_mask = torch.tensor(edge_candidate_mask).bool().to(self.device)
        # eg_loss = self.nll_loss(dgl_graphs, nx_graphs, batch["eg_edges_to_predict"])
        eg_loss = self.mdn_loss(dgl_graphs, nx_graphs, batch["eg_edges_to_predict"])

        # tt_loss = nn.CrossEntropyLoss()(dgl_graph.ndata["logits"], dgl_graph.ndata["type"])
        tt_loss = 0
        td_loss = 0

        loss = self.loss_weight_td * td_loss + self.loss_weight_eg * eg_loss + self.loss_weight_tt * tt_loss

        log = {"train_loss": loss, "td_loss": td_loss, "eg_loss": eg_loss, "tt_loss": tt_loss}

        return {"loss": loss, "log": log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5, weight_decay=0)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=BioNLPDataset.collate_fn
        )
        return loader

    def validation_step(self, batch, batch_idx):
        fname, text, ann = batch[0]
        return {fname: self.predict(text, ann, fname)}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end( self, outputs ):
        return self.validation_epoch_end(outputs)


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

        sentences = []
        for i in aggregated_outputs.values():
            sentences += i[1]
        # result, _ = self.node_detector.evaluate(sentences)
        sentence: Sentence
        for sentence in sentences[:10]:
            print(sentence.to_tagged_string())
        # log["f1_nd"] = result.main_score

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

    def mdn_sample_edge_types(self, graph, nx_graph):
        edge_types = {}

        coeff_logits = graph.edata["mixture_coefficients"].sum(dim=0)
        coeffs = torch.softmax(coeff_logits, dim=0).cpu().detach()
        component = np.random.choice(np.arange(len(coeffs)), p=coeffs.numpy(), size=1)
        # component = 0
        edge_probs = graph.edata["probs"][:, component, ...].squeeze(1).cpu().detach()
        for prob, (u, v, d) in zip(edge_probs, nx_graph.edges(data=True)):
            if d["type"] == "Candidate":
                edge_type_id = np.random.choice(np.arange(len(prob)), p=prob.numpy(), size=1)[0]
                edge_types[(u,v)] = self.id_to_edge_type[edge_type_id]
        # print(coeff_logits[2] - max(coeff_logits[:2].max(), coeff_logits[3:].max()))

        return edge_types


        # TODO sample edges from component

    def predict(self, text, ann, fname, return_batches=False):
        a1_lines = ann.a1_lines
        sentences = self.train_dataset.sentence_splitter.split(text)
        self.node_detector.predict(sentences, embedding_storage_mode="cpu")

        a2_lines = [l.strip() for l in ann.a2_lines if l.startswith("T")]
        # a2_lines = get_event_trigger_lines_from_sentences(sentences, len(a1_lines))
        ann = StandoffAnnotation(a1_lines, a2_lines)
        predicted_graph = ann.event_graph.copy()
        batches = []
        for sentence in sentences:
            entity_triggers, event_triggers = get_triggers(sentence, ann)
            trigger_spans = get_trigger_spans(entity_triggers + event_triggers, sentence)
            event_trigger_ids = set(i.id for i in event_triggers)

            for i_trigger, marked_trigger in enumerate(self.train_dataset.trigger_ordering(event_triggers, ann)):
                for i_generated in range(self.max_events_per_trigger):
                    ann = StandoffAnnotation(a1_lines, a2_lines)
                    # if marked_trigger.id not in set(i.id for i in ann.event_triggers):
                    #     continue

                    events_in_sentence = {e.id for e in ann.events.values() if e.trigger.id in event_trigger_ids}

                    event_id = get_free_event_id(predicted_graph)
                    batch = {"eg_sentence": [sentence],
                             "eg_trigger": [marked_trigger.id],
                             "eg_graph": [get_partial_graph(ann, events_in_sentence,
                                                            triggers=entity_triggers + event_triggers,
                                                            trigger_spans = trigger_spans)],
                             "eg_event_id": [event_id]}
                    batches.append(batch)

                    nx_graphs, dgl_graphs = self.forward(batch)
                    dgl_graph = dgl_graphs[0]
                    nx_graph = nx_graphs[0]

                    edge_types = self.mdn_sample_edge_types(dgl_graph, nx_graph)

                    if all(i == "None" for i in edge_types.values()):
                        break
                    else:
                        event_node_idx = list(nx_graphs[0].nodes).index(event_id)
                        # event_type = self.id_to_node_type[dgl_graph.ndata["logits"][event_node_idx].argmax().item()]
                        # if event_type not in self.train_dataset.EVENT_TYPES:
                        #     event_type = "None"
                        event_type = marked_trigger.type
                        predicted_graph.add_node(event_id, type=event_type)

                        if predicted_graph.nodes[marked_trigger.id]["type"] == "None" or predicted_graph.nodes[marked_trigger.id]["type"] == event_type:
                            predicted_graph.add_edge(marked_trigger.id, event_id, type="Trigger")
                            predicted_graph.nodes[marked_trigger.id]["type"] = event_type
                        else:
                            new_trigger_id = get_free_trigger_id(predicted_graph)
                            predicted_graph.add_node(new_trigger_id, type=event_type,
                                                     orig_span=(marked_trigger.start, marked_trigger.end),
                                                     text=marked_trigger.text
                                                     )
                            predicted_graph.add_edge(new_trigger_id, event_id, type="Trigger")

                        for (u, v), edge_type in edge_types.items():
                            if edge_type != "None":
                                predicted_graph.add_edge(u, v, type=edge_type)

                        self.clean_up_graph(predicted_graph, remove_unlifted=False, remove_invalid=False)
                        a2_lines = get_a2_lines_from_graph(predicted_graph, event_types=self.train_dataset.EVENT_TYPES)

        self.clean_up_graph(predicted_graph, remove_unlifted=True, remove_invalid=True)
        a2_lines = get_a2_lines_from_graph(predicted_graph, event_types=self.train_dataset.EVENT_TYPES)

        if return_batches:
            return "\n".join(a2_lines), batches
        else:
            return "\n".join(a2_lines), sentences



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
