from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizerFast, BertConfig

from events import consts
from events.dataset import (
    PC13Dataset,
    get_text_encoding_and_node_spans,
    get_triggers,
    get_trigger_to_position,
    get_event_linearization,
    get_event_graph,
    get_adjacency_matrix, MAX_LEN, BioNLPDataset)
from events.evaluation import Evaluator
from events.modeling_bert import BertGNNModel
from events.parse_standoff import parse_lines, StandoffAnnotation

BERTS = {"bert": BertModel, "gnn": BertGNNModel}


def lift_event_edges(a2_lines):
    trigger_to_event_id = defaultdict(list)

    for line in a2_lines:
        if not line.startswith("E"):
            continue
        fields = line.strip().split("\t")
        event_id = fields[0]
        trigger = fields[1].split()[0].split(":")[1]
        trigger_to_event_id[trigger].append(event_id)

    new_a2_lines = []
    for line in a2_lines:
        if not line.startswith("E"):
            new_a2_lines.append(line)
            continue
        fields = line.strip().split("\t")
        args = fields[-1].split()
        new_args = [args[0]]  # the event trigger

        for arg in args[1:]:
            arg_type, arg_trigger = arg.split(":")
            if arg_trigger in trigger_to_event_id:  # arg is event
                for event_id in trigger_to_event_id[arg_trigger]:
                    new_args.append(f"{arg_type}:{event_id}")
            else:  # arg is entity
                new_args.append(f"{arg_type}:{arg_trigger}")
        new_args = " ".join(new_args)
        new_a2_lines.append("\t".join(fields[:-1] + [new_args]))

    return new_a2_lines


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

        self.train_path = config["train"]
        self.dev_path = config["dev"]
        self.linearize_events = config["linearize"]

        self.dropout = nn.Dropout(0.2)

        self.train_dataset = PC13Dataset(
            Path(config["train"]),
            config["bert"],
            linearize_events=self.linearize_events,
        )
        self.dev_dataset = PC13Dataset(
            Path(config["dev"]), config["bert"], linearize_events=self.linearize_events,
            predict=True
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
            self.train_dataset, batch_size=64, shuffle=True, collate_fn=BioNLPDataset.collate_fn
        )
        return loader

    def validation_step(self, batch, batch_idx):
        fname, text, ann = batch[0]
        return {fname: self.predict(text, ann)}

    def validation_epoch_end(self, outputs):
        aggregated_outputs = {}
        for i in outputs:
            aggregated_outputs.update(i)

        evaluator = Evaluator(
            eval_script=consts.PC13_EVAL_SCRIPT,
            data_dir=self.dev_path,
            out_dir=Path("tmp/val_out"),
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
        a2_lines = [l.strip() for l in ann.a2_lines if l.startswith("T")]
        ann = StandoffAnnotation(a1_lines, a2_lines)
        n_generated_total = 0
        for sentence in self.train_dataset.sentence_splitter.split(text):
            entity_triggers, event_triggers = get_triggers(sentence, ann)
            trigger_to_position = get_trigger_to_position(sentence, ann)
            events_in_sentence = []

            for i_trigger, marked_trigger in enumerate(event_triggers):
                for i_generated in range(self.max_events_per_trigger):
                    ann = StandoffAnnotation(a1_lines, a2_lines)

                    event_to_trigger = {}
                    for event in ann.events.values():
                        try:
                            event_to_trigger[event.id] = event.trigger.id
                        except AttributeError:
                            event_to_trigger[event.id] = event.trigger.id

                    if self.linearize_events:
                        (
                            encoding_graph,
                            node_types_graph,
                            node_spans_graph,
                            node_ids_graph,
                        ) = get_event_linearization(ann=ann,
                                                    tokenizer=self.train_dataset.tokenizer,
                                                    node_type_to_id=self.train_dataset.node_type_to_id,
                                                    known_events=events_in_sentence)
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
                                            known_events=events_in_sentence)

                    remaining_length = MAX_LEN - len(encoding_graph["input_ids"])
                    if remaining_length <= 0:
                        print("Graph is too large for MAX_LENGTH. Skipping...")
                        continue
                    node_spans_graph = [(a + remaining_length, b + remaining_length) for a,b in node_spans_graph] # because we will append them to the text encoding
                    encoding_text, node_spans_text, node_types_text = get_text_encoding_and_node_spans(
                        text=str(sentence),
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
                        event_id = f"E{n_generated_total + 1}"
                        events_in_sentence.append(event_id)
                        args = []
                        args_text = []
                        for edge_type, edge_trigger in zip(
                            edge_types, entity_triggers + event_triggers
                        ):
                            if edge_type == "None":
                                continue
                            args.append(f"{edge_type}:{edge_trigger}")
                            args_text.append(ann.triggers[edge_trigger].text)
                        if args:
                            a2_lines.append(
                                f"{event_id}\t{ann.triggers[marked_trigger].type}:{marked_trigger} {' '.join(args)}"
                            )
                            n_generated_total += 1

        a2_lines = lift_event_edges(a2_lines)

        return "\n".join(a2_lines)
