import itertools
import logging
import math
import re
from bisect import bisect_right, bisect_left
from collections import defaultdict, deque
from operator import attrgetter, itemgetter
from pathlib import Path

import networkx as nx
from bioc import BioCJSONReader, BioCXMLReader
from flair.data import Sentence, Token
import transformers
import torch
from flair.tokenization import SciSpacySentenceSplitter, SegtokSentenceSplitter
from indra.statements import Statement, Agent, BoundCondition, ModCondition
from indra import statements
from indra.assemblers.english.assembler import _assemble_agent_str, EnglishAssembler
from tqdm import tqdm
from lxml import etree

from events import consts
from events.parse_standoff import StandoffAnnotation
from events import parse_standoff
from util.utils import overlaps, get_id

N_CROSS_SENTENCE = 0

MAX_LEN = 412


def get_adjacency_matrix(event_graph, nodes_text, nodes_graph, event_to_trigger,
                         edge_type_to_id):
    """
    Computes typed adjacency matrix for text + graph transformer

    Adjacency matrix is decomposed into two parts:

    `text_to_graph':  contains an edge from token to a graph node if they share the same trigger
    `graph_to_graph':  adjacency matrix of the graph => only edges from graph node to graph node
    """
    text1_trigger_ids = []
    for i in nodes_text:
        try:
            text1_trigger_ids.append(i.id)
        except AttributeError:
            text1_trigger_ids.append(i)

    node_ids_graph = []
    trigger_ids_graph = []
    for i in nodes_graph:
        try:
            node_id = {i.id}
        except AttributeError:
            node_id = i

        if not isinstance(node_id, set):
            node_id = {node_id}
        if list(node_id)[0].startswith(
                "E"):  # This is an event and thus there's only a single node in the set => Replace with trigger
            trigger_id = {event_to_trigger[list(node_id)[0]]}
        else:
            trigger_id = node_id
        node_ids_graph.append(node_id)
        trigger_ids_graph.append(trigger_id)

    text_to_graph = torch.zeros((len(text1_trigger_ids), len(node_ids_graph)))
    graph_to_graph = torch.zeros((len(node_ids_graph), len(node_ids_graph)))
    text_to_graph[:] = edge_type_to_id["None"]
    graph_to_graph[:] = edge_type_to_id["None"]

    # Text to graph
    for i, u_trigger in enumerate(text1_trigger_ids):
        for j, v_trigger in enumerate(trigger_ids_graph):
            if not isinstance(v_trigger, set):
                v_trigger = set(v_trigger)
            if u_trigger in v_trigger:  # This gets an edge if the token is the trigger for any of the nodes in the set
                text_to_graph[i, j] = edge_type_to_id["Trigger"]

    # Graph to graph
    for i, us in enumerate(node_ids_graph):
        for j, vs in enumerate(node_ids_graph):
            if not isinstance(us, set):
                us = set(us)
            if not isinstance(vs, set):
                vs = set(vs)
            for u in us:
                for v in vs:
                    if (u, v) in event_graph.edges:
                        data = event_graph.get_edge_data(u, v)
                        graph_to_graph[i, j] = edge_type_to_id[data["type"]]

    return {"text_to_graph": text_to_graph,
            "graph_to_graph": graph_to_graph}


def get_event_graph(entities, ann, tokenizer, node_type_to_id, known_events=None):
    text = ""
    node_char_spans = []
    node_types = []
    node_spans = []
    node_ids = []

    entity_text_to_pos = {}
    node_char_spans = []

    if known_events is not None:
        events = [ann.events[e] for e in known_events]
    else:
        events = ann.events.values()
    events = sorted(events, key=attrgetter("trigger.start"))

    for entity in entities:
        entity = ann.triggers[entity]
        if entity.text not in entity_text_to_pos:
            start = len(text) + 1 if text else 0
            if text:
                text += " " + entity.text
            else:
                text = entity.text
            entity_text_to_pos[entity.text] = len(entity_text_to_pos)
            node_types.append(entity.type)
            node_ids.append({entity.id})
            node_char_spans.append((start, start + len(entity.text)))
        else:
            node_ids[entity_text_to_pos[entity.text]].add(entity.id)

    for event in events:
        start = len(text) + 1
        node_char_spans.append((start, start + len(event.trigger.text)))
        node_types.append(event.trigger.type)
        node_ids.append({event.id})
        text += " " + event.trigger.text

    text += "[SEP]"
    encoding = tokenizer.encode_plus(text, return_offsets_mapping=True,
                                     add_special_tokens=False)
    token_starts = [i[0] for i in encoding["offset_mapping"]][1:-1]
    node_type_tensor = torch.zeros(len(encoding["input_ids"]))
    node_type_tensor[:] = node_type_to_id["None"]
    for (start, end), node_type in zip(node_char_spans, node_types):
        start, end = adapt_span(start=start, end=end, token_starts=token_starts)
        node_type_tensor[start:end] = node_type_to_id[node_type]
        start += 1  # because adapt_span assumes there's an unhandled [CLS] token at the start
        end += 1
        node_spans.append((start, end))

    return encoding, node_type_tensor, node_spans, node_ids


def get_events(sent, ann):
    events = []
    for event_id, event in ann.events.items():
        if sent.end_pos > int(event.trigger.start) >= sent.start_pos:
            events.append(event)

    return events


def get_trigger_to_position(sent, graph):
    trigger_to_position = {}
    for n, d in graph.nodes(data=True):
        if n.startswith("T"):
            trigger_to_position[n] = (int(d["span"][0]) - sent.start_pos,
                                      int(d["span"][1]) - sent.start_pos)
    return trigger_to_position


def get_text_encoding_and_node_spans(text, trigger_pos, tokenizer, max_length,
                                     graph, node_type_to_id, trigger_to_position,
                                     return_token_starts=False, known_triggers=None):
    if trigger_pos:
        marker_start, marker_end = trigger_pos
        marked_text = text[:marker_start] + "@ " + text[
                                                   marker_start:marker_end] + " @" + text[
                                                                                     marker_end:]
    else:
        marked_text = text
        marker_start = None
        marker_end = None
    encoding_text = tokenizer.encode_plus(marked_text, return_offsets_mapping=True,
                                          max_length=max_length, add_special_tokens=True,
                                          pad_to_max_length=True, truncation=True)

    if 0 not in encoding_text['input_ids']:
        raise ValueError("Truncated")

    token_starts = [i for i, _
                    in tokenizer.encode_plus(text, return_offsets_mapping=True,
                                             add_special_tokens=True)["offset_mapping"]]
    node_spans = get_trigger_spans(triggers=sorted(trigger_to_position),
                                   token_starts=token_starts[1:-1],
                                   marker_start=marker_start, marker_end=marker_end,
                                   trigger_to_position=trigger_to_position, text=text,
                                   encoding_text=encoding_text, tokenizer=tokenizer)
    node_types_text = torch.zeros(len(encoding_text["input_ids"]))
    node_types_text[:] = node_type_to_id["None"]
    for node, span in zip(sorted(trigger_to_position), node_spans):
        if known_triggers is None or node in known_triggers:
            if graph.nodes[node]["type"] != "Entity":
                node_types_text[span[0]:span[1]] = node_type_to_id[
                    graph.nodes[node]["type"]]

    if not return_token_starts:
        return encoding_text, node_spans, node_types_text
    else:
        return encoding_text, node_spans, node_types_text, token_starts


def adapt_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """
    start = int(start)
    end = int(end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return new_start, new_end


def get_trigger_spans(triggers, token_starts, marker_start, marker_end,
                      trigger_to_position,
                      text=None, tokenizer=None, encoding_text=None):
    trigger_spans = []

    for node in triggers:
        char_start, char_end = trigger_to_position[node]
        token_start, token_end = adapt_span(start=char_start, end=char_end,
                                            token_starts=token_starts)
        # account for [CLS]
        token_start += 1
        token_end += 1

        # adjust for inserted markers
        if marker_start is not None and char_start >= marker_start:
            token_start += 1
        if marker_end is not None and char_start >= marker_end:
            token_start += 1

        if marker_start is not None and char_end > marker_start:
            token_end += 1
        if marker_end is not None and char_end > marker_end:
            token_end += 1

        if token_start >= len(
                encoding_text["input_ids"]):  # default to [SEP] if sentence is too long
            token_start = len(encoding_text["input_ids"]) - 1
            token_end = len(encoding_text["input_ids"])

        # if text and tokenizer and encoding_text:
        #     orig_text = text[char_start:char_end].lower().replace(" ", "")
        #     marked_text = tokenizer.decode(encoding_text["input_ids"][token_start:token_end]).lower().replace(" ", "")
        #     if not orig_text == marked_text :
        #         print(orig_text, marked_text)

        trigger_spans.append((token_start, token_end))

    return trigger_spans


def triggers_are_overlapping(trigger1, trigger2):
    return max(int(trigger1.start), int(trigger2.start)) <= min(int(trigger1.end),
                                                                int(trigger2.end))


def integrate_predicted_a2_triggers(ann, ann_predicted):
    new_graph = ann.event_graph.copy()
    new_a2_lines = []
    added_triggers = set()
    original_triggers = set(e.trigger for e in ann.events.values())

    for predicted_trigger in ann_predicted.triggers.values():
        predicted_trigger.id = None  # only overlapping triggers receive an id for now

    for original_trigger in original_triggers:
        for predicted_trigger in ann_predicted.triggers.values():
            if triggers_are_overlapping(original_trigger, predicted_trigger):
                if original_trigger.id not in added_triggers:
                    predicted_trigger.id = original_trigger.id
                    new_a2_lines.append(predicted_trigger.to_a_star())
                    added_triggers.add(original_trigger.id)
                break
        else:  # didn't find an overlap
            for _, event in ann.event_graph.out_edges(original_trigger.id):
                new_graph.remove_node(event)

    for predicted_trigger in ann_predicted.triggers.values():
        if not predicted_trigger.id:
            predicted_trigger.id = get_free_trigger_id(new_graph)
            new_a2_lines.append(predicted_trigger.to_a_star())

    new_a2_lines += get_a2_lines_from_graph(new_graph)

    new_ann = StandoffAnnotation(ann.a1_lines, new_a2_lines)

    return new_ann


def filter_graph_to_sentence(text_graph, sent):
    global N_CROSS_SENTENCE

    event_to_n_edges = {i: len(text_graph.out_edges(i)) for i in text_graph.nodes if
                        i.startswith("E")}

    text_graph = text_graph.copy()
    for n, d in [n for n in text_graph.nodes(data=True)]:
        if n.startswith("T") and not (
                sent.end_pos > int(d["span"][0]) >= sent.start_pos):
            for u, _, d in [e for e in text_graph.in_edges(n, data=True)]:
                if d["type"] == "Trigger":
                    text_graph.remove_node(u)
            text_graph.remove_node(n)

    for i in text_graph.nodes:
        if i.startswith("E"):
            if len(text_graph.out_edges(i)) < event_to_n_edges[i]:
                N_CROSS_SENTENCE += 1

    return text_graph


def event_in_graph(event, graph):
    trigger_span = (event.trigger.start, event.trigger.end)
    for event_cand in [i for i in graph.nodes if i.startswith("E")]:
        if graph.nodes[event_cand]["type"] != event.type:
            continue

        trigger_cand = [i for _, i, d in graph.out_edges(event_cand, data=True) if
                        d["type"] == "Trigger"][0]

        if not overlaps(graph.nodes[trigger_cand]["span"],
                        trigger_span):
            continue

        matched_roles = []
        for role_type, role in event.roles:
            try:
                role_pos = (role.start, role.end)
            except AttributeError:
                role_pos = (role.trigger.start, role.trigger.end)

            for _, role_cand, d in graph.out_edges(event_cand, data=True):
                if overlaps(role_pos, graph.nodes[role_cand]["span"]) and d[
                    "type"] == role_type:
                    matched_roles.append(role)
                    break

        if len(matched_roles) == len(event.roles):
            return True

    return False


class BioNLPDataset:
    EVENT_TYPES = None
    ENTITY_TYPES = None
    EDGE_TYPES = None
    DUPLICATES_ALLOWED = None
    NO_ARGUMENT_ALLOWED = None
    NO_THEME_ALLOWED = None
    EVENT_TYPE_TO_ORDER = None
    EDGE_TYPES_TO_MOD = None
    EVENT_MODS = None

    def is_valid_argument_type(self, arg, reftype):
        raise NotImplementedError

    def print_example(self, example):
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        tags1 = [id_to_label[i.item()] for i in example["edge_labels"]][1:]
        tags2 = [id_to_label[i.item()] for i in example["trigger_labels"]][1:]
        tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"].tolist(),
                                                      skip_special_tokens=True)
        sentence = Sentence()
        for token, tag1, tag2 in zip(tokens, tags1, tags2):
            token = Token(token.replace("##", ""))
            sentence.add_token(token)
            sentence.tokens[-1].add_label("Edge", tag1)
            sentence.tokens[-1].add_label("Trigger", tag2)
        print(sentence.to_tagged_string())

    def get_event_linearization(self, graph, tokenizer, node_type_to_id,
                                edge_types_to_mod, event_ordering, known_events=None,
                                known_triggers=None):
        linearization = ""
        node_char_spans = []
        node_types = []
        node_spans = []
        node_ids = []

        events = event_ordering(graph)

        if known_events is not None:
            events = [e for e in events if e in known_events]

        for event in events:
            edge_type_to_trigger = defaultdict(set)
            for u, v, data in graph.out_edges(event, data=True):
                edge_type_to_trigger[data["type"]].add(v)
            try:
                trigger_id = [v for u, v, d in graph.out_edges(event, data=True) if
                              d["type"] == "Trigger"][0]
            except IndexError:
                trigger_id = None

            # add Cause
            if "Cause" in edge_type_to_trigger:
                for i, trigger in enumerate(edge_type_to_trigger["Cause"]):
                    linearization = self.add_text_to_linearization(
                        graph=graph,
                        node_id=trigger,
                        linearization=linearization,
                        node_char_spans=node_char_spans,
                        node_types=node_types,
                        node_ids=node_ids,
                        known_nodes=known_triggers
                    )
                    if i > 0:
                        linearization += " and"

                linearization += " causes"

            # if trigger_id is not None:
            # add trigger (with Theme)
            linearization = self.add_text_to_linearization(
                graph=graph,
                node_id=event,
                linearization=linearization,
                node_char_spans=node_char_spans,
                node_types=node_types,
                node_ids=node_ids,
                known_nodes=known_triggers
            )

            if "Theme" in edge_type_to_trigger:
                themes = sorted(edge_type_to_trigger["Theme"])

                linearization += " of"

                trigger = themes[0]
                linearization = self.add_text_to_linearization(
                    graph=graph,
                    node_id=trigger,
                    linearization=linearization,
                    node_char_spans=node_char_spans,
                    node_types=node_types,
                    node_ids=node_ids,
                    known_nodes=known_triggers
                )

                for trigger in themes[1:]:
                    linearization += " and"
                    linearization = self.add_text_to_linearization(
                        graph=graph,
                        node_id=trigger,
                        linearization=linearization,
                        node_char_spans=node_char_spans,
                        node_types=node_types,
                        node_ids=node_ids,
                        known_nodes=known_triggers
                    )

            # add rest
            for edge_type, mod in sorted(edge_types_to_mod.items()):
                if edge_type in edge_type_to_trigger:
                    linearization += " " + mod
                    triggers = sorted(edge_type_to_trigger[edge_type])

                    trigger = triggers[0]
                    linearization = self.add_text_to_linearization(
                        graph=graph,
                        node_id=trigger,
                        linearization=linearization,
                        node_char_spans=node_char_spans,
                        node_types=node_types,
                        node_ids=node_ids,
                        known_nodes=known_triggers
                    )

                    for trigger in triggers[1:]:
                        linearization += " and"

                        linearization = self.add_text_to_linearization(
                            graph=graph,
                            node_id=trigger,
                            linearization=linearization,
                            node_char_spans=node_char_spans,
                            node_types=node_types,
                            node_ids=node_ids,
                            known_nodes=known_triggers
                        )

            linearization += " |"
        linearization += "[SEP]"

        encoding = tokenizer.encode_plus(linearization, return_offsets_mapping=True,
                                         add_special_tokens=False)
        token_starts = [i[0] for i in encoding["offset_mapping"]][:-1]
        node_type_tensor = torch.zeros(len(encoding["input_ids"]))
        node_type_tensor[:] = node_type_to_id["None"]
        for (start, end), node_type in zip(node_char_spans, node_types):
            start, end = adapt_span(start=start, end=end, token_starts=token_starts)
            node_type_tensor[start:end] = node_type_to_id[node_type]
            node_spans.append((start, end))

        return encoding, node_type_tensor, node_spans, node_ids

    def add_text_to_linearization(self, graph, linearization, node_char_spans,
                                  node_types, node_ids,
                                  node_id, known_nodes):

        if graph.nodes[node_id]["type"] in self.EVENT_TYPES:
            text = graph.nodes[node_id]["type"]
        else:
            text = graph.nodes[node_id]["text"]
        if known_nodes is None or node_id in known_nodes and graph.nodes[node_id][
            "type"] != "Entity":
            node_type = graph.nodes[node_id]["type"]
        else:
            node_type = "None"
        start = len(linearization) + 1
        end = start + len(text)
        node_char_spans.append((start, end))
        node_ids.append(node_id)
        node_types.append(node_type)
        linearization += " " + text

        return linearization

    @staticmethod
    def collate(batch):
        assert len(batch) == 1
        return batch[0]

    def __init__(self, path: Path, tokenizer: Path,
                 linearize_events: bool = False, batch_size: int = 16,
                 predict: bool = False,
                 predict_entities: bool = False, event_order: str = "position",
                 max_span_width: int = 10, small=False
                 ):
        self.text_files = [f for f in path.glob('*.txt')]
        if small:
            # self.text_files = [i for i in self.text_files if "PMID-10593988" in str(i)]
            self.text_files = self.text_files[23:24]
        # self.text_files = self.text_files[226:227]

        self.node_type_to_id = {}
        for i in sorted(itertools.chain(self.EVENT_TYPES, self.ENTITY_TYPES)):
            if i not in self.node_type_to_id:
                self.node_type_to_id[i] = len(self.node_type_to_id)

        self.edge_type_to_id = {v: i for i, v in enumerate(sorted(self.EDGE_TYPES))}
        self.trigger_to_id = {v: i for i, v in enumerate(sorted(self.EVENT_TYPES))}
        self.label_to_id = {"O": 0}
        for edge_type in sorted(self.EDGE_TYPES) + sorted(self.EVENT_TYPES):
            if "B-" + edge_type not in self.label_to_id:
                self.label_to_id["B-" + edge_type] = len(self.label_to_id)
            if "I-" + edge_type not in self.label_to_id:
                self.label_to_id["I-" + edge_type] = len(self.label_to_id)
        self.event_mod_to_id = {v: i for i, v in enumerate(sorted(self.EVENT_MODS))}

        if small:
            self.sentence_splitter = SegtokSentenceSplitter()
            self.batch_size = 2
        else:
            self.sentence_splitter = SciSpacySentenceSplitter()
            self.batch_size = batch_size
        self.predict = predict
        self.linearize_events = linearize_events
        self.predict_entities = predict_entities
        self.max_span_width = max_span_width

        if event_order == "id":
            self.event_ordering = self.sort_events_by_id
        elif event_order == "position":
            self.event_ordering = self.sort_events_by_position
        elif event_order == "simple_first":
            self.event_ordering = self.sort_events_by_simple_first
        else:
            raise ValueError(event_order)

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(str(tokenizer))
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["@"]})

        self.examples = []
        self.predicted_examples = []
        self.predict_example_by_fname = {}
        self.ann_by_fname = {}

        for file in tqdm(self.text_files, desc="Parsing raw data"):
            if file.with_suffix(".a2").exists():
                with file.with_suffix(".a2").open() as f:
                    a2_lines = f.readlines()
            else:
                a2_lines = []
            with file.open() as f, file.with_suffix(".a1").open() as f_a1:
                text = f.read()
                a1_lines = f_a1.readlines()
                ann = StandoffAnnotation(a1_lines=a1_lines, a2_lines=a2_lines)
                self.examples += self.generate_examples(text, ann)
                self.predict_example_by_fname[file.name] = (file.name, text, ann)
                self.ann_by_fname[file.name] = ann

        self.fnames = sorted(self.predict_example_by_fname)
        self.print_and_reset_statistics()

    def print_and_reset_statistics(self):
        print(f"{N_CROSS_SENTENCE}/{len(self.examples)} cross sentence")
        print(f"{parse_standoff.N_SELF_LOOPS}/{len(self.examples)} self loops")

    def get_triggers(self, sent, graph):
        entity_triggers = []
        event_triggers = []
        for n, d in graph.nodes(data=True):
            if n.startswith("T"):
                if not (sent.end_pos > int(d["span"][0]) >= sent.start_pos):
                    continue

                if d["type"] in self.EVENT_TYPES:
                    event_triggers.append(n)
                elif d["type"] in self.ENTITY_TYPES:
                    entity_triggers.append(n)
                else:
                    raise ValueError((n, d["type"]))

        return entity_triggers, event_triggers

    def sort_events_by_position(self, text_graph):
        n_max_roles = 10

        sorted_events = []
        sort_tuple_to_events = defaultdict(list)

        for n, d in text_graph.nodes(data=True):
            if n.startswith("E") or n.startswith("Fail"):
                roles = []
                role_starts = []
                role_types = []
                trigger_start = None
                for _, v, d in text_graph.out_edges(n, data=True):
                    if d["type"] == "Trigger":
                        trigger_start = int(text_graph.nodes[v]['span'][0])
                        trigger_type = text_graph.nodes[v]['type']
                    else:
                        roles.append(
                            (int(text_graph.nodes[v]['span'][0]),
                             text_graph.nodes[v]['type']))
                for i in range(n_max_roles - len(roles)):
                    roles.append((0, "AAAA"))
                for role_start, role_type in sorted(roles):
                    role_starts.append(role_start)
                    role_types.append(role_type)

                if trigger_start is None:  # this is a Fail sort by rightmost role
                    trigger_start = sorted([i[0] for i in roles])[-1]
                    trigger_type = "Fail"

                sort_tuple = tuple(
                    [trigger_start] + role_starts + [trigger_type] + role_types)
                sort_tuple_to_events[sort_tuple].append(n)

        for sort_tuple in sorted(sort_tuple_to_events):
            sorted_events += sort_tuple_to_events[sort_tuple]

        return sorted_events

    def sort_events_by_id(self, graph):
        events = [n for n in graph.nodes if n.startswith("E")]
        return sorted(events)

    def sort_events_by_simple_first(self, G):
        order = []
        n_max_roles = 10

        G = G.copy()
        self.lift_event_edges(G)
        for u in list(G.nodes()):
            if not u.startswith("E"):
                G.remove_node(u)
        zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        while zero_outdegree:
            sort_tuple_to_events = defaultdict(list)
            sort_tuple = []
            for event in zero_outdegree:
                sort_tuple.append(G.nodes[event]["type"])
                sort_tuple_to_events[tuple(sort_tuple)].append(event)
                G.remove_node(event)

            for sort_tuple in sorted(sort_tuple_to_events):
                order += sort_tuple_to_events[sort_tuple]

            zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        return order

    def sort_triggers_by_simple_first(self, triggers, ann):
        triggers_with_info = []
        for trigger in triggers:
            triggers_with_info.append((trigger,
                                       ann.triggers[trigger].start,
                                       self.EVENT_TYPE_TO_ORDER[
                                           ann.triggers[trigger].type]))

        sorted_triggers = sorted(triggers_with_info, key=itemgetter(2, 1))

        return [i[0] for i in sorted_triggers]

    def __getitem__(self, item):

        if self.predict:
            fname = self.fnames[item]
            return self.predict_example_by_fname[fname]
        else:
            example = self.examples[item]

            return example

    def __len__(self):
        if self.predict:
            return len(self.fnames)
        else:
            return len(self.examples)

    def get_overlapping_triggers(self, trigger, graph):
        overlapping_triggers = []

        for n, d in graph.nodes(data=True):
            if not n.startswith("T"):
                continue
            if overlaps((d["span"]), graph.nodes[trigger]["span"]):
                overlapping_triggers.append(n)

        return overlapping_triggers

    def split_args(self, graph):
        for event in [n for n in graph.nodes if n.startswith("E")]:
            singular_edges_by_type = defaultdict(set)
            valid_edges = []
            event_type = graph.nodes[event]["type"]
            for u, v, edge_data in graph.out_edges(event, data=True):
                edge_type = edge_data["type"]
                edge = u, v, edge_type
                if (event_type, edge_type) not in self.DUPLICATES_ALLOWED:
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
                    graph.add_node(new_event, **graph.nodes[event])

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

            triggers = [v for _, v, d in graph.out_edges(event, data=True) if
                        d["type"] == "Trigger"]
            if not edge_types or edge_types == ["Trigger"]:
                if event_type not in self.NO_ARGUMENT_ALLOWED:
                    graph.remove_node(event)
                    continue

            elif "Theme" not in edge_types and event_type not in self.NO_THEME_ALLOWED:
                graph.remove_node(event)
                continue

            elif not triggers:
                graph.remove_node(event)
                continue

    def remove_invalid_edges(self, graph):
        # graph.remove_edges_from(nx.selfloop_edges(graph))
        for event in [n for n in graph.nodes if n.startswith("E")]:
            event_type = graph.nodes[event]["type"]
            for u, v, key, d in list(graph.out_edges(event, data=True, keys=True)):
                edge_type = d["type"]
                v_type = graph.nodes[v]["type"]
                if not self.is_valid_argument_type(event_type=event_type, arg=edge_type,
                                                   reftype=v_type, refid=v):
                    graph.remove_edge(u, v, key)

    def remove_nones(self, graph):
        for node in [n for n in graph.nodes]:
            if graph.nodes[node]["type"] == "None":
                graph.remove_node(node)

    def clean_up_graph(self, graph: nx.DiGraph, remove_invalid=False, lift=False,
                       allow_entity=False):
        old_nodes = None
        for i in range(10):
            if old_nodes == list(graph.nodes()):
                break
            old_nodes = list(graph.nodes())

            if lift:
                self.lift_event_edges(graph, allow_entity=allow_entity)

            if remove_invalid:
                self.remove_invalid_edges(graph)
            if lift:  # there shouldn't be any cycles left after lifting
                self.break_up_cycles(graph)
            self.split_args(graph)

            if remove_invalid:
                self.remove_nones(graph)
                self.remove_invalid_events(graph)

    def break_up_cycles(self, graph):
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

    def lift_event_edges(self, graph, allow_entity=False):
        for trigger in [n for n in graph.nodes if n.startswith("T")]:
            if graph.nodes[trigger]["type"] in self.EVENT_TYPES:
                events = []
                for t in self.get_overlapping_triggers(trigger, graph):
                    for u, _, d in graph.in_edges(t, data=True):
                        if d["type"] == "Trigger":
                            events.append(u)
                for u, _, edge_id, d in list(
                        graph.in_edges(trigger, data=True, keys=True)):
                    if d["type"] != "Trigger":
                        for event in events:
                            if not u.startswith("Fail"):
                                u_trigger = \
                                    [i for _, i, d in graph.out_edges(u, data=True) if
                                     d["type"] == "Trigger"][0]
                            else:
                                u_trigger = u
                            event_trigger = \
                                [i for _, i, d in graph.out_edges(event, data=True) if
                                 d["type"] == "Trigger"][0]
                            if u_trigger != event_trigger:
                                graph.add_edge(u, event, **d)
                        if events or not allow_entity:
                            graph.remove_edge(u, trigger, edge_id)
                        else:
                            graph.nodes[trigger]["type"] = "Entity"

    @property
    def n_event_types(self):
        return len(self.EVENT_TYPES)

    @property
    def n_node_types(self):
        return len(consts.NODE_TYPES)

    @property
    def n_edge_types(self):
        return len(consts.EDGE_TYPES)

    @property
    def n_entity_types(self):
        return len(self.ENTITY_TYPES)

    def generate_examples(self, text, ann):
        trigger_to_events = defaultdict(list)
        event_to_trigger = {}
        for event in ann.events.values():
            try:
                trigger_to_events[event.trigger.id].append(event)
                event_to_trigger[event.id] = event.trigger.id
            except AttributeError:
                trigger_to_events[event.trigger].append(event)
                event_to_trigger[event.id] = event.trigger.id

        examples = []

        for sentence in self.sentence_splitter.split(text):
            graph = ann.text_graph.copy()

            graph = filter_graph_to_sentence(ann.text_graph, sentence)
            n_edge_before = len(graph.edges)
            self.clean_up_graph(graph, remove_invalid=True)
            # n_edge_after = len(graph.edges)
            # if n_edge_after  < n_edge_before:
            #     print(n_edge_before - n_edge_after)
            entity_triggers, event_triggers = self.get_triggers(sentence, graph)
            trigger_to_position = get_trigger_to_position(sentence, graph)

            known_events = []

            for event in self.event_ordering(graph):
                example = self.build_example(ann=ann,
                                             entity_triggers=entity_triggers,
                                             event=event,
                                             event_triggers=event_triggers,
                                             graph=graph,
                                             sentence=sentence,
                                             trigger_to_position=trigger_to_position,
                                             known_events=known_events)

                if example is not None:
                    examples.append(example)

                known_events.append(event)

            # signal end of generation
            example = self.build_example(ann=ann,
                                         entity_triggers=entity_triggers,
                                         event=None,
                                         event_triggers=event_triggers,
                                         graph=graph,
                                         sentence=sentence,
                                         trigger_to_position=trigger_to_position,
                                         known_events=known_events)
            if example is not None:
                examples.append(example)

        # for example in examples:
        #     self.print_example(example)
        return examples

    def build_example(self, ann, entity_triggers, event,
                      event_triggers, graph, sentence,
                      trigger_to_position, known_events, graph_to_encode=None,
                      ):
        example = {}

        known_triggers = set(
            entity_triggers + [ann.events[i].trigger.id for i in known_events])
        if graph_to_encode is not None:
            encoding_graph, node_types_graph = self.get_event_linearization(
                graph=graph_to_encode, edge_types_to_mod=self.EDGE_TYPES_TO_MOD,
                tokenizer=self.tokenizer, node_type_to_id=self.node_type_to_id,
                event_ordering=self.event_ordering,
                known_triggers=None, known_events=None)[
                                               :2]  # this is a fully predicted graph and thus we know everything about it
        else:
            encoding_graph, node_types_graph = self.get_event_linearization(
                graph=graph, edge_types_to_mod=self.EDGE_TYPES_TO_MOD,
                known_events=known_events, known_triggers=known_triggers,
                tokenizer=self.tokenizer, node_type_to_id=self.node_type_to_id,
                event_ordering=self.event_ordering)[:2]

        remaining_length = MAX_LEN - len(encoding_graph["input_ids"])
        trigger_to_position = get_trigger_to_position(sentence, graph)

        try:
            encoding_text, node_spans_text, node_types_text = get_text_encoding_and_node_spans(
                text=sentence.to_original_text(),
                tokenizer=self.tokenizer,
                max_length=remaining_length,
                node_type_to_id=self.node_type_to_id,
                trigger_to_position=trigger_to_position,
                trigger_pos=None,
                graph=graph,
                known_triggers=known_triggers
            )
        except:
            return None

        input_ids = torch.cat([torch.tensor(encoding_text["input_ids"]),
                               torch.tensor(encoding_graph["input_ids"])])
        token_type_ids = torch.zeros(input_ids.size(0))
        token_type_ids[len(encoding_text["input_ids"]):] = 1
        node_types = torch.cat([node_types_text, node_types_graph])

        trigger_to_span = {}
        for trigger, span in zip(sorted(trigger_to_position), node_spans_text):
            trigger_to_span[trigger] = span

        edge_labels = torch.zeros_like(input_ids)
        trigger_labels = torch.zeros_like(input_ids)
        edge_labels[:] = self.label_to_id["O"]
        trigger_labels[:] = self.label_to_id["O"]
        mod_labels = torch.zeros(len(self.event_mod_to_id))
        if event:
            event_type = graph.nodes[event]["type"]
            for u, v, data in graph.out_edges(event, data=True):
                if v.startswith("E"):
                    raise ValueError(
                        "Text graph should only have Event -> Trigger edges")

                if v in trigger_to_span:
                    pos = trigger_to_span[v]
                    if data["type"] == "Trigger":
                        trigger_labels[pos[0]] = self.label_to_id["B-" + event_type]
                        trigger_labels[pos[0] + 1: pos[1]] = self.label_to_id[
                            "I-" + event_type]
                    else:
                        edge_labels[pos[0]] = self.label_to_id["B-" + data["type"]]
                        edge_labels[pos[0] + 1: pos[1]] = self.label_to_id[
                            "I-" + data["type"]]

            for mod in graph.nodes[event]["modifications"]:
                mod_labels[self.event_mod_to_id[mod]] = 1

            trigger_label = torch.tensor(self.trigger_to_id[event_type])
        else:
            trigger_label = torch.tensor(self.trigger_to_id["None"])

        example["input_ids"] = input_ids.long()
        example["token_type_ids"] = token_type_ids.long()
        example["edge_labels"] = edge_labels.long()
        example["trigger_labels"] = trigger_labels.long()
        example["trigger_label"] = trigger_label.long()
        example["mod_labels"] = mod_labels.long()
        example["node_type_ids"] = node_types.long()

        id_to_node_type = {v: k for k, v in self.node_type_to_id.items()}

        foo = []
        for tok, nt in zip(self.tokenizer.convert_ids_to_tokens(input_ids.tolist()),
                           node_types):
            foo.append((tok, id_to_node_type[nt.item()]))
        # print(foo[:20])
        # print(foo[-20:])

        return example

    @staticmethod
    def collate_fn(examples):
        keys_to_batch = {"input_ids", "token_type_ids", "attention_mask",
                         "node_type_ids",
                         "mod_labels", "trigger_label"}
        batch = defaultdict(list)
        for example in examples:
            for k, v in example.items():
                batch[k].append(v)

        batched_batch = {}
        for k, v in list(batch.items()):
            for i in keys_to_batch:
                if i in k:
                    batched_batch[k] = torch.stack(v)
                    break
            else:
                batched_batch[k] = v

        return batched_batch

    def add_dagger_examples(self, examples, dry_run=False):
        n_added = 0
        existing_trajectories = set(self.example_to_trajectory(i) for i in self.examples)
        for pred_example in examples:
            ann_true = self.ann_by_fname[pred_example["fname"][0]]

            sentence = pred_example["sentence"][0]
            graph_true = filter_graph_to_sentence(ann_true.text_graph, sentence)
            # known_events = []
            for event in self.event_ordering(graph_true):
                if not event_in_graph(ann_true.events[event], pred_example["graph"]):
                    example = self.build_dagger_example(pred_example=pred_example,
                                                        event=event,
                                                        true_graph=graph_true,
                                                        sentence=sentence)
                    break
            else:
                example = self.build_dagger_example(pred_example=pred_example,
                                                    event=None, true_graph=graph_true,
                                                    sentence=sentence)

            if self.example_to_trajectory(example) not in existing_trajectories:
                n_added += 1
                if not dry_run:
                    self.examples.append(example)
                # self.print_example(example)

        logging.info(f"Added {n_added} new examples from Dagger")

    def example_to_trajectory(self, example):
        tokens = "".join(
            self.tokenizer.convert_ids_to_tokens(example["input_ids"].tolist(),
                                                 skip_special_tokens=True))
        edge_labels = tuple(example["edge_labels"].tolist())
        trigger_labels = tuple(example["trigger_labels"].tolist())

        return (tokens, edge_labels, trigger_labels)

    def build_dagger_example(self, pred_example, event, true_graph, sentence):
        example = {}

        trigger_to_position = get_trigger_to_position(sentence, true_graph)
        _, node_spans_text, _ = get_text_encoding_and_node_spans(
            text=sentence.to_original_text(),
            tokenizer=self.tokenizer,
            max_length=MAX_LEN,
            node_type_to_id=self.node_type_to_id,
            trigger_to_position=trigger_to_position,
            trigger_pos=None,
            graph=true_graph,
            known_triggers=None
        )

        input_ids = pred_example["input_ids"][0]
        edge_labels = torch.zeros_like(input_ids)
        trigger_labels = torch.zeros_like(input_ids)
        edge_labels[:] = self.label_to_id["O"]
        trigger_labels[:] = self.label_to_id["O"]
        mod_labels = torch.zeros(len(self.event_mod_to_id))
        trigger_to_span = {}

        for trigger, span in zip(sorted(trigger_to_position), node_spans_text):
            trigger_to_span[trigger] = span

        if event is not None:
            for u, v, data in true_graph.out_edges(event, data=True):
                event_type = true_graph.nodes[event]["type"]
                if v.startswith("E"):
                    raise ValueError(
                        "Text graph should only have Event -> Trigger edges")

                if v in trigger_to_span:
                    pos = trigger_to_span[v]
                    if data["type"] == "Trigger":
                        trigger_labels[pos[0]] = self.label_to_id["B-" + event_type]
                        trigger_labels[pos[0] + 1: pos[1]] = self.label_to_id[
                            "I-" + event_type]
                    else:
                        edge_labels[pos[0]] = self.label_to_id["B-" + data["type"]]
                        edge_labels[pos[0] + 1: pos[1]] = self.label_to_id[
                            "I-" + data["type"]]

            for mod in true_graph.nodes[event]["modifications"]:
                mod_labels[self.event_mod_to_id[mod]] = 1

        example["input_ids"] = input_ids.long()
        example["token_type_ids"] = pred_example['token_type_ids'][0].long()
        example["edge_labels"] = edge_labels.long()
        example["trigger_labels"] = trigger_labels.long()
        example["mod_labels"] = mod_labels.long()
        example["node_type_ids"] = pred_example['node_type_ids'][0].long()

        return example


class PC13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.PC13_EVENT_TYPES
    ENTITY_TYPES = consts.PC13_ENTITY_TYPES
    EDGE_TYPES = consts.PC13_EDGE_TYPES
    DUPLICATES_ALLOWED = consts.PC13_DUPLICATES_ALLOWED
    NO_THEME_ALLOWED = consts.PC13_NO_THEME_ALLOWED
    NO_ARGUMENT_ALLOWED = consts.PC13_NO_ARGUMENT_ALLOWED
    EVENT_TYPE_TO_ORDER = consts.PC13_EVENT_TYPE_TO_ORDER
    EDGE_TYPES_TO_MOD = consts.PC13_EDGE_TYPES_TO_MOD
    EVENT_MODS = consts.PC13_EVENT_MODS

    def __init__(self, path: Path, bert_path: Path,
                 batch_size: int = 16, predict=False, event_order="position",
                 linearize_events: bool = False, small=False):
        super().__init__(path, bert_path, batch_size=batch_size, predict=predict,
                         linearize_events=linearize_events, event_order=event_order,
                         small=small
                         )

    def is_valid_argument_type(self, event_type, arg, reftype, refid):
        if arg == "Trigger":
            return reftype in self.EVENT_TYPES

        if event_type == "Binding":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Transport":
            if arg in {"Cause", "Product"}:
                return False

        if "activation" in event_type.lower():
            if arg in {"AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Transcription":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Gene_expression":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Conversion":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc",
                       "Participant"}:
                return False

        if event_type == "Degradation":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Dissociation":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Pathway":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Localization":
            if arg in {"Cause"}:
                return False

        if "acetylation" in event_type.lower():
            if arg in {"AtLoc", "ToLoc", "Loc", "FromLoc", "Product"}:
                return False

        if "regulation" in event_type.lower():
            if arg in {"AtLoc", "Site", "ToLoc", "Loc", "FromLoc", "Participant"}:
                return False

        if "activation" in event_type.lower():
            if arg in {"AtLoc", "Site", "ToLoc", "Loc", "FromLoc", "Participant"}:
                return False

        if "regulation" not in event_type.lower() and reftype not in self.ENTITY_TYPES:
            return False

        if "loc" in arg.lower():
            if reftype not in {"Entity", "Cellular_component"}:
                return False

        return True


def get_free_event_id(graph):
    ids = [int(n[1:]) for n in graph.nodes if n.startswith("E")]
    max_id = max(ids) if ids else 0
    free_id = f"E{max_id + 1}"
    return free_id


def get_free_trigger_id(graph):
    ids = [int(n[1:]) for n in graph.nodes if n.startswith("T")]
    max_id = max(ids) if ids else 0
    free_id = f"T{max_id + 1}"
    return free_id


def get_free_fail_id(graph):
    ids = [int(n[4:]) for n in graph.nodes if n.startswith("Fail")]
    max_id = max(ids) if ids else 0
    free_id = f"Fail{max_id + 1}"
    return free_id


def sort_args(args):
    order = ["Theme", "Theme2", "Theme3", "Theme4", "Theme5", "Cause", "Site", "CSite",
             "AtLoc", "ToLoc"]

    def key(x):
        if x in order:
            return order.index(x)
        else:
            return len(order)

    return sorted(args, key=key)


def get_a2_lines_from_graph(graph: nx.DiGraph, event_types):
    lines = []
    n_mods = 0

    for trigger, d in graph.nodes(data=True):
        if trigger.startswith("T") and d["type"] in event_types or d["type"] == "Entity":
            lines.append(
                f"{trigger}\t{graph.nodes[trigger]['type']} {' '.join(graph.nodes[trigger]['span'])}\t{graph.nodes[trigger]['text']}")

    for event in [n for n in graph.nodes if n.startswith("E")]:
        try:
            trigger = [v for u, v, d in graph.out_edges(event, data=True) if
                       d["type"] == "Trigger"][0]
        except IndexError:
            __import__("ipdb").set_trace()
        event_type = graph.nodes[event]["type"]
        args = []
        edge_type_count = defaultdict(int)
        for _, v, d in graph.out_edges(event, data=True):
            edge_type = d['type']
            if edge_type == "Trigger":
                continue
            edge_type_count[edge_type] += 1
            if edge_type_count[edge_type] == 1:
                args.append(f"{edge_type}:{v}")
            else:
                args.append(
                    f"{edge_type}{edge_type_count[edge_type]}:{v}")  # for Theme2, Product2 etc.
        args = sort_args(args)
        lines.append(f"{event}\t{event_type}:{trigger} {' '.join(args)}")

        for mod in graph.nodes[event]["modifications"]:
            n_mods += 1
            lines.append(f"M{n_mods}\t{mod} {event}")

    return lines


class INDRADataset:
    def __init__(self, path, tokenizer):
        reader = BioCJSONReader(path)
        reader.read()
        self.documents = reader.collection.documents
        self.node_type_to_id, self.edge_type_to_id = self._get_dicts()
        self.label_to_id = {"O": 0}
        for edge_type in self.edge_type_to_id:
            self.label_to_id["B-" + edge_type] = len(self.label_to_id)
            self.label_to_id["I-" + edge_type] = len(self.label_to_id)
        for node_type in self.node_type_to_id:
            self.label_to_id[node_type] = len(self.label_to_id)

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(tokenizer)
        self.examples = self._get_examples()

    def _get_dicts(self):
        node_type_to_id = {}
        edge_type_to_id = {}
        for document in self.documents:
            stmts = []
            for passage in document.passages:
                for stmt_json in passage.infons["indra"]:
                    stmt = Statement._from_json(stmt_json)
                    stmts.append(stmt)
            G_full = self.stmts_to_graph(stmts)
            for _, d in G_full.nodes(data=True):
                if d["type"] not in node_type_to_id:
                    node_type_to_id[d["type"]] = len(node_type_to_id)
            for _, _, d in G_full.edges(data=True):
                if d["type"] not in edge_type_to_id:
                    edge_type_to_id[d["type"]] = len(edge_type_to_id)

        return node_type_to_id, edge_type_to_id

    def _get_examples(self):
        examples = []
        for document in self.documents:
            for passage in document.passages:
                stmts = []
                text1 = passage.text
                for stmt_json in passage.infons["indra"]:
                    stmt = Statement._from_json(stmt_json)
                    stmts.append(stmt)

                G_full = self.stmts_to_graph(stmts)

                node_to_pos = defaultdict(list)
                for ann in passage.annotations:
                    ann_id = (ann.infons["type"], ann.infons["id"])
                    found_node = None
                    for n, d in G_full.nodes(data=True):
                        if "db_refs" in d:
                            db_refs = [tuple(i) for i in d["db_refs"].items()]
                            if ann_id in db_refs:
                                found_node = n
                                break

                    if found_node:
                        for loc in ann.locations:
                            node_to_pos[found_node].append((int(loc.offset),
                                                            int(loc.offset) + int(
                                                                loc.length)))

                known_nodes = [u for u, d in G_full.nodes(data=True) if
                               d["type"] == "entity"]
                for node in self.node_order(G_full):
                    if G_full.nodes[node]["type"] != "entity":
                        example = self._build_example(node, G_full=G_full,
                                                      known_nodes=known_nodes,
                                                      text1=text1,
                                                      node_to_pos1=node_to_pos)
                        known_nodes.append(node)
                        examples.append(example)

        return examples

    def stmts_to_graph(self, stmts):
        G = nx.MultiDiGraph()
        agent_to_node = {}
        for i, stmt in enumerate(stmts):
            stmt_id = get_id()
            G.add_node(stmt_id, type="stmt_" + stmt.__class__.__name__)
            for ag_name in stmt._agent_order:
                agents = getattr(stmt, ag_name)
                if not isinstance(agents, list):
                    agents = [agents]
                for i, agent in enumerate(agents):
                    if agent.matches_key() not in agent_to_node:
                        if agent.bound_conditions:
                            node, G_agent = self.bound_agent_to_graph(agent)
                            G.add_nodes_from(G_agent.nodes(data=True))
                            G.add_edges_from(G_agent.edges(data=True))
                        elif agent.mods:
                            node, G_agent = self.modified_agent_to_graph(agent)
                            G.add_nodes_from(G_agent.nodes(data=True))
                            G.add_edges_from(G_agent.edges(data=True))
                        else:
                            G.add_node(agent.name, type="entity", db_refs=agent.db_refs)
                            node = agent.name
                        agent_to_node[agent.matches_key()] = node
                    node = agent_to_node[agent.matches_key()]
                    G.add_edge(stmt_id, node, type=ag_name)

            try:
                if stmt.residue and stmt.position:
                    site = stmt.residue + stmt.position
                    G.add_node(site, db_refs={"RESIDUE": site}, type="entity")
                    G.add_edge(stmt_id, site, type="site")
            except AttributeError:
                pass

        G_new = nx.DiGraph()
        G_new.add_nodes_from(G.nodes(data=True))
        edge_to_types = defaultdict(set)
        for u, v, d in G.edges(data=True):
            edge_to_types[(u, v)].add(d["type"])
        for (u, v), types in edge_to_types.items():
            combined_type = ",".join(sorted(types))
            G_new.add_edge(u, v, type=combined_type)

        return G_new

    def bound_agent_to_graph(self, agent):
        G = nx.MultiDiGraph()
        complex_id = get_id()
        G.add_node(complex_id, type="complex")
        if agent.mods:
            new_agent, G_agent = self.modified_agent_to_graph(agent)
            G.add_nodes_from(G_agent.nodes(data=True))
            G.add_edges_from(G_agent.edges(data=True))
        else:
            G.add_node(agent.name, db_refs=agent.db_refs, type="entity")
            new_agent = agent.name

        G.add_edge(complex_id, new_agent, type="has_member")
        for bc in agent.bound_conditions:
            if bc.agent.mods:
                new_agent, G_agent = self.modified_agent_to_graph(bc.agent)
                G.add_nodes_from(G_agent.nodes(data=True))
                G.add_edges_from(G_agent.edges(data=True))
            else:
                G.add_node(bc.agent.name, db_refs=bc.agent.db_refs, type="entity")
                new_agent = bc.agent.name
            G.add_edge(complex_id, new_agent, type="has_member")

        return complex_id, G

    def modified_agent_to_graph(self, agent):
        G = nx.MultiDiGraph()
        mod_agent_id = get_id()
        G.add_node(mod_agent_id, type="modified_entity")
        G.add_node(agent.name, type="entity", db_refs=agent.db_refs)
        G.add_edge(mod_agent_id, agent.name, type="theme")
        for mod in agent.mods:
            site = mod.residue + mod.position
            G.add_node(site, db_refs={"RESIDUE": site}, type="entity")
            G.add_edge(mod_agent_id, site, type=mod.mod_type + "_at")

        return mod_agent_id, G

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

    def _build_example(self, node, G_full, known_nodes, text1, node_to_pos1):
        G_known = G_full.subgraph(known_nodes)
        text2, node_to_pos2 = self.linearize_graph(G_known)
        # for u, positions in node_to_pos2.items():
        #     node_to_pos2[u] = [(s+len(text1), e+len(text1)) for s, e in positions]
        encoding = self.tokenizer.encode_plus(text1, text2, padding="max_length",
                                              truncation="only_first",
                                              return_tensors="pt",
                                              return_offsets_mapping=True,
                                              max_length=MAX_LEN)
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        seq2_start = torch.where(encoding["offset_mapping"].sum(dim=1) == 0)[0][1].item()
        pad_start = torch.where(encoding["offset_mapping"].sum(dim=1) == 0)[0][2].item()
        token_starts1 = encoding["offset_mapping"][:seq2_start, 0].tolist()
        token_starts2 = encoding["offset_mapping"][seq2_start:pad_start, 0].tolist()
        labels = torch.zeros_like(encoding["input_ids"])
        labels[0] = self.label_to_id[
            G_full.nodes[node]["type"]]  # Predict node type with [CLS]
        for _, v, d in G_full.out_edges(node, data=True):
            if v in node_to_pos1:
                node_to_pos = node_to_pos1
                token_starts = token_starts1
                offset = 0
            else:
                node_to_pos = node_to_pos2
                token_starts = token_starts2
                offset = seq2_start

            for start, end in node_to_pos[v]:
                span = adapt_span(start, end, token_starts)
                labels[span[0] + offset] = self.label_to_id["B-" + d["type"]]
                for i in range(span[0] + 1 + offset, span[1] + offset):
                    labels[i] = self.label_to_id["I-" + d["type"]]

        encoding["labels"] = labels

        return encoding

    def node_order(self, G):
        G = G.copy()
        order = []
        zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        while zero_outdegree:
            sort_tuple_to_events = defaultdict(list)
            sort_tuple = []
            for node in zero_outdegree:
                sort_tuple.append(G.nodes[node]["type"])
                sort_tuple_to_events[tuple(sort_tuple)].append(node)
                G.remove_node(node)

            for sort_tuple in sorted(sort_tuple_to_events):
                order += sort_tuple_to_events[sort_tuple]

            zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        return order

    def linearize_graph(self, G):
        linearization = ""
        node_to_pos = {}

        node_to_stmt = self.graph_to_statements(G)
        for node in self.node_order(G):
            node_type = G.nodes[node]["type"]
            if node in node_to_stmt:
                if node_type.startswith("stmt"):
                    text = EnglishAssembler([node_to_stmt[node]]).make_model()
                else:
                    text = _assemble_agent_str(node_to_stmt[node]).agent_str + "."
                start = len(linearization)
                linearization += text + " "
                end = len(linearization) - 1
                node_to_pos[node] = [(start, end)]

        return linearization, node_to_pos

    def graph_to_statements(self, G: nx.MultiDiGraph):
        def subgraph_to_statement(n):
            kwargs = {}
            for _, v, d in G.out_edges(n, data=True):
                agent = subgraph_to_agent(v)
                types = d["type"].split(",")
                for t in types:
                    if t == "site":
                        kwargs["residue"] = v[0]
                        kwargs["position"] = v[1:]
                        continue
                    if t in kwargs:
                        if not isinstance(kwargs[t], list):
                            kwargs[t] = [kwargs[t]]
                        kwargs[t].append(agent)
                    else:
                        kwargs[t] = agent

            stmt_type = G.nodes[n]["type"][len("stmt_"):]
            stmt = getattr(statements, stmt_type)(**kwargs)
            return stmt

        def subgraph_to_agent(n):
            if G.nodes[n]["type"] == "complex":
                members = []
                for _, v, d in G.out_edges(n, data=True):
                    if d["type"] == "has_member":
                        members.append(subgraph_to_agent(v))
                agent = members[0]
                for i in members[1:]:
                    agent.bound_conditions.append(BoundCondition(i))

            elif G.nodes[n]["type"] == "modified_entity":
                theme = \
                [v for _, v, d in G.out_edges(n, data=True) if d["type"] == "theme"][0]
                agent = subgraph_to_agent(theme)
                for _, v, d in G.out_edges(n, data=True):
                    if d["type"].endswith("_at"):
                        mod_type = d["type"][:-len("_at")]
                        pos = v[1:]
                        res = v[:1]
                        agent.mods.append(
                            ModCondition(mod_type, position=pos, residue=res))
            elif G.nodes[n]["type"] == "entity":
                agent = Agent(n)
                agent.db_refs = G.nodes[n]["db_refs"]
            else:
                raise ValueError(G.nodes[n]["type"])

            return agent

        node_to_stmt = {}

        nodes = []
        for n, d in G.nodes(data=True):
            if d["type"] != "entity":
                nodes.append(n)

        for n in nodes:
            if G.nodes[n]["type"].startswith("stmt_"):
                node_to_stmt[n] = subgraph_to_statement(n)
            else:
                node_to_stmt[n] = subgraph_to_agent(n)

        return node_to_stmt

    def print_example(self, example):
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        tags = [id_to_label[i.item()] for i in example["labels"]]
        tokens = ["CLS"] + self.tokenizer.convert_ids_to_tokens(
            example["input_ids"].tolist(),
            skip_special_tokens=True)
        sentence = Sentence()
        for token, tag in zip(tokens, tags):
            token = Token(token.replace("##", ""))
            sentence.add_token(token)
            sentence.tokens[-1].add_label("Label", tag)
        print(sentence.to_tagged_string())

    def example_to_brat(self, example):
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        tags = [id_to_label[i.item()] for i in example["labels"]]
        tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"].tolist(),
                                                      skip_special_tokens=False)
        tokens = [i for i in tokens if i != "[PAD]"]
        sentence = Sentence()
        for token, tag in zip(tokens, tags):
            whitespace_before = False
            if "##" in token:
                token = token.replace("##", "")
            else:
                whitespace_before = True
            start = len(sentence.to_original_text()) + 1
            if not whitespace_before:
                start -= 1
            sentence.add_token(Token(token, start_position=start))
            sentence.tokens[-1].add_label("Label", tag)
        a1_lines = []
        for i, span in enumerate(sentence.get_spans("Label")):
            mention = sentence.to_original_text()[span.start_pos:span.end_pos]
            a1_lines.append(
                f"T{i}\t{span.tag} {span.start_pos} {span.end_pos}\t{mention}")

        return sentence.to_original_text(), "\n".join(a1_lines)


class BELDataset:
    supported_functions = {"complexAbundance", "proteinModification", "degradation",
                           "translocation", "molecularActivity"}
    supported_relations = {"increases", "directlyIncreases", "decreases",
                           "directlyDecreases", "-|", "=|", "->", "=>"}
    relation_to_simplified = {
        "increases": "increases",
        "directlyIncreases": "increases",
        "->": "increases",
        "=>": "increases",
        "decreases": "decreases",
        "directlyDecreases": "decreases",
        "-|": "decreases",
        "=|": "decreases"
    }

    function_type_to_shortened = {
        "abundance": "a",
        "activity": "act",
        "biologicalProcess": "bp",
        "cellSecretion": "sec",
        "cellSurfaceExpression": "surf",
        "complexAbundance": "complex",
        "compositeAbundance": "composite",
        "degradation": "deg",
        "fragment": "frag",
        "fusion": "fus",
        "geneAbundance": "g",
        "location": "loc",
        "microRNAAbundance": "m",
        "molecularActivity": "ma",
        "pathology": "path",
        "proteinAbundance": "p",
        "proteinModification": "pmod",
        "reaction": "rxn",
        "rnaAbundance": "r",
        "translocation": "tloc",
        "variant": "var",
    }

    def __init__(self, path, tokenizer, simplify_activations=True,
                 remove_graphs_with_unsupported_functions=True,
                 simplify_relations=True):
        self.simplify_activations = simplify_activations
        self.remove_graphs_with_unsupported_functions = remove_graphs_with_unsupported_functions
        self.simplify_relations = simplify_relations
        text_to_graphs = self.read_bioc(path)
        self.text_to_graph = {}
        for text, graphs in text_to_graphs.items():
            graphs = self.rename_and_filter_graphs(graphs)
            if graphs:
                G = self.merge_graphs(graphs)
                self.text_to_graph[text] = G
        self.node_type_to_id, self.edge_type_to_id = self._get_dicts()
        self.label_to_id = {'O': 0,
                            'B-self': 1,
                            'I-self': 2,
                            'B-cause': 3,
                            'I-cause': 4,
                            'B-theme': 5,
                            'I-theme': 6,
                            'B-member': 7,
                            'I-member': 8,
                            'proteinAbundance': 9,
                            'entity': 10,
                            'increases': 11,
                            'molecularActivity': 12,
                            'abundance': 13,
                            'rnaAbundance': 14,
                            'biologicalProcess': 15,
                            'complexAbundance': 16,
                            'degradation': 17,
                            'pathology': 18,
                            'decreases': 19,
                            'translocation': 20}

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(tokenizer)
        self.examples = self._get_examples()

    def _get_dicts(self):
        node_type_to_id = {}
        edge_type_to_id = {}
        for i, G in enumerate(self.text_to_graph.values()):
            for _, d in G.nodes(data=True):
                if d["type"] not in node_type_to_id:
                    node_type_to_id[d["type"]] = len(node_type_to_id)
            for _, _, d in G.edges(data=True):
                if d["type"] not in edge_type_to_id:
                    edge_type_to_id[d["type"]] = len(edge_type_to_id)

        return node_type_to_id, edge_type_to_id

    def merge_graphs(self, graphs):
        G_merged = nx.DiGraph()
        for graph in graphs:
            G_merged.add_nodes_from(graph.nodes(data=True))
            G_merged.add_edges_from(graph.edges(data=True))
        return G_merged

    def rename_and_filter_graphs(self, graphs):
        filtered_graphs = []
        for G in graphs:
            G_new = nx.DiGraph()
            for u, v, d in G.edges(data=True):
                name_u = self.node_to_bel(u, G)
                name_v = self.node_to_bel(v, G)
                e = (name_u, name_v)

                if e in G_new.edges and d["type"] != G_new.edges[e][
                    "type"]:  # remove self-loops
                    print("removed: ", e)
                    print(d["type"], G_new.edges[e]["type"])
                    print()

                    break

                G_new.add_node(name_u, **G.nodes[u])
                G_new.add_node(name_v, **G.nodes[v])
                G_new.add_edge(name_u, name_v, **d)
            else:
                filtered_graphs.append(G_new)

        return filtered_graphs

    def example_to_sentence(self, input_ids, labels):
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        tokens = ["CLS"] + self.tokenizer.convert_ids_to_tokens(input_ids.tolist(),
                                                                skip_special_tokens=True)
        tags = [id_to_label[i.item()] for i in labels]
        sentence = Sentence()
        for token, tag in zip(tokens, tags):
            token = Token(token.replace("##", ""))
            sentence.add_token(token)
            sentence.tokens[-1].add_label("Label", tag)

        return sentence

    def read_bioc(self, path):
        tree = etree.parse(str(path))
        text_to_graphs = defaultdict(list)
        for document in tqdm(list(tree.xpath('//document'))):
            for passage in document.xpath("./passage"):
                G = nx.DiGraph()
                text = passage.xpath("./text")[0].text
                for ann in passage.xpath("./annotation"):
                    type = ann.xpath("./infon[@key='type']")[0].text
                    if type in {"relationship", "ModificationArgument"}:
                        continue

                    infon_dict = {}
                    for infon in ann.xpath("./infon"):
                        infon_dict[infon.attrib["key"]] = infon.text

                    if "GOCCID" in infon_dict:  # location labeled as gene due to bug
                        continue

                    locations = ann.xpath("./location")
                    if locations:
                        infon_dict["span"] = (int(locations[0].attrib["start"]),
                                              int(locations[0].attrib["start"]) + int(
                                                  locations[0].attrib["offset"]))
                        infon_dict["text"] = ann.xpath("./text")[0].text

                    for namespace in ["EGID", "HGNC", "MGI", "CHEBI", "GOBP", "MESHD"]:
                        if namespace in infon_dict:
                            name = infon_dict[namespace]
                            break
                    else:
                        raise ValueError(etree.dump(ann))
                    G.add_node(name, span=infon_dict["span"], type="entity",
                               namespace=namespace, text=infon_dict["text"])
                    del infon_dict["span"]
                    G.add_node(ann.attrib["id"], **infon_dict)
                    G.add_edge(ann.attrib["id"], name, type="self")

                for rel in passage.xpath("./relation"):
                    infon_dict = {}

                    type = rel.xpath("./infon[@key='type']")[0].text
                    if type == "proteinModification":
                        mod_type_ref = \
                        rel.xpath("./node[@role='ModificationType']")[0].attrib["refid"]
                        mod_type = passage.xpath(
                            f"./annotation[@id='{mod_type_ref}']/infon[@key='ModificationType']")[
                            0].text
                        infon_dict["type"] = type + "_" + mod_type
                    elif "activity" in type.lower() and self.simplify_activations:
                        infon_dict["type"] = "molecularActivity"
                    elif type in self.supported_relations:
                        if self.simplify_relations:
                            infon_dict["type"] = self.relation_to_simplified[type]
                        else:
                            infon_dict["type"] = type

                    elif type not in self.supported_functions and self.remove_graphs_with_unsupported_functions:
                        G = None  # signals graph should be removed
                        break
                    else:
                        infon_dict["type"] = type

                    for infon in rel.xpath("./infon"):
                        if not infon.attrib[
                                   "key"] == "type":  # already processed type above
                            infon_dict[infon.attrib["key"]] = infon.text
                    if type == "translocation":  # fix translocation annotation bug and remove locations
                        theme = rel.xpath("./node")[0]
                        theme.attrib['role'] = 'theme'
                        for loc in rel.xpath('./node')[1:]:
                            rel.remove(loc)
                    if G:
                        G.add_node(rel.attrib["id"], **infon_dict)
                        for node in rel.xpath("./node"):
                            if not node.attrib["role"] == "relationship":
                                G.add_edge(rel.attrib["id"], node.attrib["refid"],
                                           type=node.attrib["role"])
                if G:
                    text_to_graphs[text].append(G)

        return text_to_graphs

    def _build_example(self, node, G_full, known_nodes, text1):
        G_known = G_full.subgraph(known_nodes)
        text2 = self.linearize_graph(G_known)
        encoding = self.tokenizer.encode_plus(text1, text2, padding="max_length",
                                              truncation="only_first",
                                              return_tensors="pt",
                                              return_offsets_mapping=True,
                                              max_length=MAX_LEN)
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        seq2_start = torch.where(encoding["offset_mapping"].sum(dim=1) == 0)[0][1].item()
        pad_start = torch.where(encoding["offset_mapping"].sum(dim=1) == 0)[0][2].item()
        token_starts1 = encoding["offset_mapping"][:seq2_start, 0].tolist()
        token_starts2 = encoding["offset_mapping"][seq2_start:pad_start, 0].tolist()
        labels = torch.zeros_like(encoding["input_ids"])

        if node is not None:
            labels[0] = self.label_to_id[
                G_full.nodes[node]["type"]]  # Predict node type with [CLS]
            for _, v, d in G_full.out_edges(node, data=True):
                if "span2" in G_full.nodes[v]:
                    span = G_full.nodes[v]["span2"]
                    token_starts = token_starts2
                    offset = seq2_start
                else:
                    span = G_full.nodes[v]["span"]
                    token_starts = token_starts1
                    offset = 0

                span = adapt_span(span[0], span[1], token_starts)
                labels[span[0] + offset] = self.label_to_id["B-" + d["type"]]
                for i in range(span[0] + 1 + offset, span[1] + offset):
                    labels[i] = self.label_to_id["I-" + d["type"]]

        encoding["labels"] = labels

        return encoding

    def linearize_graph(self, G):
        linearization = ""

        for node in self.node_order(G):
            if G.nodes[node]["type"] == "entity":
                continue
            bel = self.node_to_bel(node, G)
            start = len(linearization) + 2
            linearization += "| " + bel + " "
            end = len(linearization)
            G.nodes[node]["span2"] = (start, end)

        return linearization

    def node_to_bel(self, node, G):
        type = self.shorten_bel_type(G.nodes[node]["type"])
        if type == "entity":
            return G.nodes[node]["text"]
        elif type in self.relation_to_simplified:
            cause = \
            [i for _, i, d in G.out_edges(node, data=True) if d["type"] == "cause"][0]
            theme = \
            [i for _, i, d in G.out_edges(node, data=True) if d["type"] == "theme"][0]
            cause_bel = self.node_to_bel(cause, G)
            theme_bel = self.node_to_bel(theme, G)
            return f"{cause_bel} {type} {theme_bel}"

        else:
            args = ', '.join(self.node_to_bel(i, G) for _, i in G.out_edges(node))
            return f"{type}({args})"

    def _get_examples(self):
        examples = []
        for text, G_full in tqdm(list(self.text_to_graph.items())):
            known_nodes = [i for i, d in G_full.nodes(data=True) if
                           d["type"] == "entity"]
            for node in self.node_order(G_full):
                if G_full.nodes[node]["type"] != "entity":
                    example = self._build_example(node, G_full=G_full,
                                                  known_nodes=known_nodes,
                                                  text1=text)
                    known_nodes.append(node)
                    examples.append(example)
            example = self._build_example(None, G_full=G_full,
                                          known_nodes=known_nodes,
                                          text1=text)
            known_nodes.append(node)
            examples.append(example)

        return examples

    def node_order(self, G):
        G_orig = G
        G = G.copy()
        order = []
        zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        while zero_outdegree:
            sort_tuple_to_events = defaultdict(list)
            for node in zero_outdegree:
                sort_tuple = []
                if "span" in G.nodes[node]:
                    sort_tuple += G.nodes[node]["span"]
                else:
                    sort_tuple.append((1000000, 1000000))
                arg_starts = []
                arg_order_indices = []
                for _, arg in G_orig.out_edges(node):
                    arg_data = G_orig.nodes[arg]
                    if "span" in arg_data:
                        arg_starts += arg_data["span"]
                    try:
                        arg_order_indices.append(order.index(arg))
                    except ValueError:
                        arg_order_indices.append(1000000)

                if len(arg_starts) < 20:
                    arg_starts += [1000000] * (20 - len(arg_starts))
                if len(arg_order_indices) < 20:
                    arg_order_indices += [1000000] * (20 - len(arg_order_indices))

                sort_tuple += sorted(arg_starts)
                sort_tuple += sorted(arg_order_indices)
                sort_tuple.append(G.nodes[node]["type"])
                sort_tuple_to_events[tuple(sort_tuple)].append(node)
                G.remove_node(node)

            for sort_tuple in sorted(sort_tuple_to_events):
                order += sort_tuple_to_events[sort_tuple]

            zero_outdegree = [u for u, d in G.out_degree() if d == 0]

        return order

    def shorten_bel_type(self, bel_type):
        if bel_type in self.relation_to_simplified:
            return bel_type
        elif bel_type in self.function_type_to_shortened.values():
            return bel_type
        elif bel_type in self.function_type_to_shortened:
            return self.function_type_to_shortened[bel_type]
        elif bel_type == "entity":
            return bel_type
        else:
            raise ValueError(bel_type)

    def example_to_brat(self, example):
        id_to_label = {v: k for k, v in self.label_to_id.items()}
        tags = [id_to_label[i.item()] for i in example["labels"]]
        tokens = self.tokenizer.convert_ids_to_tokens(example["input_ids"].tolist(),
                                                      skip_special_tokens=False)
        tokens = [i for i in tokens if i != "[PAD]"]
        sentence = Sentence()
        for token, tag in zip(tokens, tags):
            whitespace_before = False
            if "##" in token:
                token = token.replace("##", "")
            else:
                whitespace_before = True
            start = len(sentence.to_original_text()) + 1
            if not whitespace_before:
                start -= 1
            sentence.add_token(Token(token, start_position=start))
            sentence.tokens[-1].add_label("Label", tag)
        a1_lines = []
        for i, span in enumerate(sentence.get_spans("Label")):
            mention = sentence.to_original_text()[span.start_pos:span.end_pos]
            a1_lines.append(
                f"T{i}\t{span.tag} {span.start_pos} {span.end_pos}\t{mention}")

        return sentence.to_original_text(), "\n".join(a1_lines)

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    ds = BELDataset("events/data/BioCreative_BEL_Track/train.bioc.xml",
                    tokenizer="/vol/fob-vol1/mi15/weberple/glusterfs/data/scibert_scivocab_uncased")
    # ds = BELDataset("foo.bioc.xml",
    #              tokenizer="/vol/fob-vol1/mi15/weberple/glusterfs/data/scibert_scivocab_uncased")
    print(len(ds))
    for i, example in enumerate(ds[:100]):
        txt, ann = ds.example_to_brat(example)
        fname = str(i).zfill(6)
        with open(f"tools/brat-v1.3_Crunchy_Frog/data/bel/{fname}.txt", "w") as f:
            f.write(txt)
        with open(f"tools/brat-v1.3_Crunchy_Frog/data/bel/{fname}.ann", "w") as f:
            f.write(ann)
