import itertools
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from flair.models import SequenceTagger
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast, BertConfig
import networkx as nx

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
    get_a2_lines_from_graph
)
from events.evaluation import Evaluator
from events.modeling_bert import BertGNNModel
from events.parse_standoff import StandoffAnnotation

BERTS = {"bert": BertModel, "gnn": BertGNNModel}


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
        return


class EventExtractor(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.train_path = config["train"]
        self.dev_path = config["dev"]
        self.linearize_events = config["linearize"]
        self.trigger_detector = SequenceTagger.load(config["trigger_detector"])
        self.output_dir = config["output_dir"]

        DatasetType = get_dataset_type(config["train"])

        self.dropout = nn.Dropout(0.2)

        self.train_dataset = DatasetType(
            Path(config["train"]),
            config["bert"],
            linearize_events=self.linearize_events,
            trigger_detector=self.trigger_detector,
            trigger_ordering=config["trigger_ordering"]
        )
        self.dev_dataset = DatasetType(
            Path(config["dev"]), config["bert"], linearize_events=self.linearize_events,
            predict=True,
            trigger_detector=self.trigger_detector,
            trigger_ordering=config["trigger_ordering"]
        )

        bert_config = BertConfig.from_pretrained(config["bert"])
        bert_config.node_type_vocab_size = len(self.train_dataset.node_type_to_id)
        bert_config.edge_type_vocab_size = len(self.train_dataset.edge_type_to_id)
        self.bert = BertGNNModel.from_pretrained(config["bert"], config=bert_config)
        # self.bert = BertModel.from_pretrained(config["bert"], config=bert_config)
        self.tokenizer = BertTokenizerFast.from_pretrained(config["bert"])

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["@"]})
        self.edge_classifier = nn.Linear(768*3, len(self.train_dataset.edge_type_to_id))
        self.loss_fn = nn.CrossEntropyLoss()
        self.max_events_per_trigger = 10

        self.id_to_edge_type = {
            v: k for k, v in self.train_dataset.edge_type_to_id.items()
        }


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
            if "Theme" not in edge_types and (set(edge_types) & self.train_dataset.NO_THEME_FORBIDDEN):
                graph.remove_node(event)

    def remove_invalid_edges(self, graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))
        for event in [n for n in graph.nodes if n.startswith("E")]:
            for u, v, d in list(graph.out_edges(event, data=True)):
                edge_type = d["type"]
                v_type = graph.nodes[v]["type"]
                if not self.train_dataset.is_valid_argument_type(edge_type, v_type):
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
        if self.linearize_events:
            attention_mask = self.get_full_attention_mask(batch["input_ids"]).long()
            edge_type_ids = None
            position_ids = None
        else:
            for i, (adjacency_matrix, node_spans_graph) in enumerate(zip(batch["adjacency_matrix"], batch["node_spans_graph"])):
                assert adjacency_matrix["text_to_graph"].shape[1] == len(node_spans_graph)
            edge_type_ids = self.adjacency_matrix_to_edge_types(
                adjacency_matrix=batch["adjacency_matrix"],
                input_ids=batch["input_ids"],
                node_spans_text=batch["node_spans_text"],
                node_spans_graph=batch["node_spans_graph"],
                sep_id=self.tokenizer.encode("")[-1]
            )
            edge_type_ids = edge_type_ids.to(self.device).long()
            # attention_mask = edge_type_ids != self.train_dataset.edge_type_to_id["None"]
            attention_mask = self.get_full_attention_mask(batch["input_ids"]).long()
            position_ids = self.get_gnn_position_ids(input_ids=batch["input_ids"],
                                                     sep_id=self.tokenizer.encode("")[-1]).long()
        node_type_ids = []
        for node_types_text, node_types_graph in zip(batch["node_types_text"], batch["node_types_graph"]):
            node_type_ids.append(torch.cat([node_types_text, node_types_graph]))
        node_type_ids = torch.stack(node_type_ids).long()

        xs, _ = self.bert(
            input_ids=batch["input_ids"].long(),
            # edge_type_ids=edge_type_ids,
            attention_mask=attention_mask,
            token_type_ids=batch["token_type_ids"].long(),
            node_type_ids=node_type_ids,
            # position_ids=position_ids
        )
        xs = self.dropout(xs)
        batch_logits = []
        for x, node_spans_text, node_spans_graph in zip(
            xs,
            batch["node_spans_text"],
            batch["node_spans_graph"]
        ):
            x_nodes = []
            for node_span in node_spans_text:
                span_rep = [x[node_span[0]],
                            torch.mean(x[node_span[0] : node_span[1]], dim=0),
                            x[node_span[1]]]
                x_nodes.append(torch.cat(span_rep))
            x_nodes = torch.stack(x_nodes)
            logits = self.edge_classifier(x_nodes)
            batch_logits.append(logits)
        return batch_logits

    def training_step(self, batch, batch_idx):
        batch_logits = self(batch)
        loss = 0
        for i, (logits, labels) in enumerate(zip(batch_logits, batch["labels"])):
            labels = torch.tensor(labels).to(self.device)
            loss += self.loss_fn(logits, labels)
        loss /= len(batch)
        log = {"train_loss": loss}
        return {"loss": loss, "log": log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-5)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, collate_fn=BioNLPDataset.collate_fn
        )
        return loader

    def validation_step(self, batch, batch_idx):
        fname, text, ann = batch[0]
        return {fname: self.predict(text, ann)}

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
        log.update(evaluator.evaluate(aggregated_outputs))

        return {"val_f1": torch.tensor(log["f1"]), "log": log}

    def val_dataloader(self):
        loader = DataLoader(
            self.dev_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x,
        )
        return loader

    def predict(self, text, ann):
        a1_lines = ann.a1_lines
        sentences = self.train_dataset.sentence_splitter.split(text)
        a2_trigger_lines = [l.strip() for l in ann.a2_lines if l.startswith("T")]
        # a2_trigger_lines = get_event_trigger_lines_from_sentences(sentences, self.trigger_detector,
        #                                        len(a1_lines))
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
                                                    known_events=event_ids_in_sentence)
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
                    current_batch["input_ids"] = torch.cat([torch.tensor(encoding_text["input_ids"]),
                                           torch.tensor(encoding_graph["input_ids"])]).unsqueeze(0).to(self.device)
                    current_batch["token_type_ids"] = torch.zeros(current_batch["input_ids"].size(0)).unsqueeze(0).to(self.device)
                    current_batch["token_type_ids"][:, len(encoding_text["input_ids"]):] = 1

                    current_batch["node_spans_text"] = torch.tensor([node_spans_text]).to(self.device)
                    current_batch["node_spans_graph"] = torch.tensor([node_spans_graph]).to(self.device)
                    current_batch["node_types_text"] = node_types_text.unsqueeze(0).to(self.device)
                    current_batch["node_types_graph"] = node_types_graph.unsqueeze(0).to(self.device)

                    current_batch["adjacency_matrix"] = [adjacency_matrix]


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

                        self.clean_up_graph(predicted_graph)
                        a2_event_lines = get_a2_lines_from_graph(predicted_graph)

        return "\n".join(a2_trigger_lines + a2_event_lines)
