import itertools
import logging
import math
import re
from bisect import bisect_right, bisect_left
from collections import defaultdict
from operator import attrgetter, itemgetter
from pathlib import Path

import networkx as nx
from flair.data import Sentence, Token
import transformers
import torch
from flair.tokenization import SciSpacySentenceSplitter, SegtokSentenceSplitter
from tqdm import tqdm

from events import consts
from events.parse_standoff import StandoffAnnotation
from events import parse_standoff

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
        if list(node_id)[0].startswith("E"): # This is an event and thus there's only a single node in the set => Replace with trigger
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
            if u_trigger in v_trigger: # This gets an edge if the token is the trigger for any of the nodes in the set
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
            start = len(text)+1 if text else 0
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
        node_char_spans.append((start, start+len(event.trigger.text)))
        node_types.append(event.trigger.type)
        node_ids.append({event.id})
        text += " " + event.trigger.text

    text += "[SEP]"
    encoding = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
    token_starts = [i[0] for i in encoding["offset_mapping"]][1:-1]
    node_type_tensor = torch.zeros(len(encoding["input_ids"]))
    node_type_tensor[:] = node_type_to_id["None"]
    for (start, end), node_type in zip(node_char_spans, node_types):
        start, end = adapt_span(start=start, end=end, token_starts=token_starts)
        node_type_tensor[start:end] = node_type_to_id[node_type]
        start += 1 # because adapt_span assumes there's an unhandled [CLS] token at the start
        end += 1
        node_spans.append((start,end))


    return encoding, node_type_tensor, node_spans, node_ids


def get_event_linearization(graph, tokenizer, node_type_to_id,
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
        trigger_id = [v for u, v, d in graph.out_edges(event, data=True) if d["type"] == "Trigger"][0]

        # add Cause
        if "Cause" in edge_type_to_trigger:
            assert len(edge_type_to_trigger["Cause"]) == 1
            trigger = list(edge_type_to_trigger["Cause"])[0]
            linearization = add_text_to_linearization(
                graph=graph,
                node_id=trigger,
                linearization=linearization,
                node_char_spans=node_char_spans,
                node_types=node_types,
                node_ids=node_ids,
                known_nodes=known_triggers
            )

            linearization += " causes"

        # add trigger (with Theme)
        linearization = add_text_to_linearization(
            graph=graph,
            node_id=trigger_id,
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
            linearization = add_text_to_linearization(
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
                linearization = add_text_to_linearization(
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
                linearization = add_text_to_linearization(
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

                    linearization = add_text_to_linearization(
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

    encoding = tokenizer.encode_plus(linearization, return_offsets_mapping=True, add_special_tokens=False)
    token_starts = [i[0] for i in encoding["offset_mapping"]][:-1]
    node_type_tensor = torch.zeros(len(encoding["input_ids"]))
    node_type_tensor[:] = node_type_to_id["None"]
    for (start, end), node_type in zip(node_char_spans, node_types):
        start, end = adapt_span(start=start, end=end, token_starts=token_starts)
        node_type_tensor[start:end] = node_type_to_id[node_type]
        node_spans.append((start,end))

    return encoding, node_type_tensor, node_spans, node_ids


def add_text_to_linearization(graph, linearization, node_char_spans, node_types, node_ids,
                              node_id, known_nodes):
    text = graph.nodes[node_id]["text"]
    if known_nodes is None or node_id in known_nodes and graph.nodes[node_id]["type"] != "Entity":
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
        marked_text = text[:marker_start] + "@ " + text[ marker_start:marker_end] + " @" + text[
                                                                               marker_end:]
    else:
        marked_text = text
        marker_start = None
        marker_end = None
    encoding_text = tokenizer.encode_plus(marked_text, return_offsets_mapping=True, max_length=max_length, add_special_tokens=True,
                                          pad_to_max_length=True, truncation=True)
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
                node_types_text[span[0]:span[1]] = node_type_to_id[graph.nodes[node]["type"]]

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


def get_trigger_spans(triggers, token_starts, marker_start, marker_end, trigger_to_position,
                      text=None, tokenizer=None, encoding_text=None):
    trigger_spans = []

    for node in triggers:
        char_start, char_end = trigger_to_position[node]
        token_start, token_end = adapt_span(start=char_start, end=char_end, token_starts=token_starts)
        # account for [CLS]
        token_start += 1
        token_end +=1

        # adjust for inserted markers
        if marker_start is not None and char_start >= marker_start:
            token_start += 1
        if marker_end is not None and char_start >= marker_end:
            token_start += 1

        if marker_start is not None and char_end > marker_start:
            token_end += 1
        if marker_end is not None and char_end > marker_end:
            token_end += 1

        if token_start >= len(encoding_text["input_ids"]): # default to [SEP] if sentence is too long
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
    return max(int(trigger1.start), int(trigger2.start)) <= min(int(trigger1.end), int(trigger2.end))


def integrate_predicted_a2_triggers(ann, ann_predicted):
    new_graph = ann.event_graph.copy()
    new_a2_lines = []
    added_triggers = set()
    original_triggers = set(e.trigger for e in ann.events.values())

    for predicted_trigger in ann_predicted.triggers.values():
        predicted_trigger.id = None # only overlapping triggers receive an id for now

    for original_trigger in original_triggers:
        for predicted_trigger in ann_predicted.triggers.values():
            if triggers_are_overlapping(original_trigger, predicted_trigger):
                if original_trigger.id not in added_triggers:
                    predicted_trigger.id = original_trigger.id
                    new_a2_lines.append(predicted_trigger.to_a_star())
                    added_triggers.add(original_trigger.id)
                break
        else: # didn't find an overlap
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

    event_to_n_edges = {i: len(text_graph.out_edges(i)) for i in text_graph.nodes if i.startswith("E")}

    text_graph = text_graph.copy()
    for n, d in [n for n in text_graph.nodes(data=True)]:
        if n.startswith("T") and not (sent.end_pos > int(d["span"][0]) >= sent.start_pos):
            for u, _, d in [e for e in text_graph.in_edges(n, data=True)]:
                if d["type"] == "Trigger":
                    text_graph.remove_node(u)
            text_graph.remove_node(n)

    for i in text_graph.nodes:
        if i.startswith("E"):
            if len(text_graph.out_edges(i)) < event_to_n_edges[i]:
                N_CROSS_SENTENCE += 1

    return text_graph




class BioNLPDataset:
    EVENT_TYPES = None
    ENTITY_TYPES = None
    EDGE_TYPES = None
    DUPLICATES_ALLOWED = None
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


    @staticmethod
    def collate(batch):
        assert len(batch) == 1
        return batch[0]

    def __init__(self, path: Path, tokenizer: Path,
                 linearize_events: bool = False, batch_size: int = 16, predict: bool = False,
                 predict_entities: bool = False, event_order: str = "position",
                 max_span_width: int = 10, small=False
                 ):
        self.text_files = [f for f in path.glob('*.txt')]
        if small:
            self.text_files = self.text_files[:2] + [i for i in self.text_files if "PMID-10085159" in str(i)]
        # self.text_files = self.text_files[:2] + [i for i in self.text_files if "PMID-10085159" in str(i)]
        # self.text_files = self.text_files[226:227]


        self.node_type_to_id = {}
        for i in sorted(itertools.chain(self.EVENT_TYPES, self.ENTITY_TYPES)):
            if i not in self.node_type_to_id:
                self.node_type_to_id[i] = len(self.node_type_to_id)

        self.edge_type_to_id = {v: i for i, v in enumerate(sorted(self.EDGE_TYPES))}
        self.label_to_id = {"O": 0}
        for edge_type in sorted(self.EDGE_TYPES) + sorted(self.EVENT_TYPES):
            if "B-" + edge_type not in self.label_to_id:
                self.label_to_id["B-" + edge_type] = len(self.label_to_id)
            if "I-" + edge_type not in self.label_to_id:
                self.label_to_id["I-" + edge_type] = len(self.label_to_id)
        self.event_mod_to_id = {v: i for i, v in enumerate(self.EVENT_MODS)}

        if small:
            self.sentence_splitter = SegtokSentenceSplitter()
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
            if n.startswith("E"):
                roles = []
                role_starts = []
                role_types = []
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

                sort_tuple = tuple([trigger_start] + role_starts + [trigger_type] + role_types)
                sort_tuple_to_events[sort_tuple].append(n)

        for sort_tuple in sorted(sort_tuple_to_events):
            sorted_events += sort_tuple_to_events[sort_tuple]

        return sorted_events

    def sort_events_by_id(self, graph):
        events = [n for n in graph.nodes if n.startswith("E")]
        return sorted(events)

    def sort_triggers_by_simple_first(self, triggers, ann):
        triggers_with_info = []
        for trigger in triggers:
            triggers_with_info.append((trigger,
                                      ann.triggers[trigger].start,
                                      self.EVENT_TYPE_TO_ORDER[ann.triggers[trigger].type]))

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
            graph = filter_graph_to_sentence(ann.text_graph, sentence)
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
            examples.append(example)

        # for example in examples:
        #     self.print_example(example)
        return examples


    def build_example(self, ann, entity_triggers, event,
                      event_triggers, graph, sentence,
                      trigger_to_position, known_events, encoding_graph=None,
                      ):
        example = {}

        known_triggers = set(entity_triggers + [ann.events[i].trigger.id for i in known_events])
        if encoding_graph is None:
            encoding_graph, node_types_graph = get_event_linearization(
                graph=graph, edge_types_to_mod=self.EDGE_TYPES_TO_MOD, known_events=known_events, known_triggers=known_triggers,
                tokenizer=self.tokenizer, node_type_to_id=self.node_type_to_id, event_ordering=self.event_ordering)[:2]

        remaining_length = MAX_LEN - len(encoding_graph["input_ids"])
        trigger_to_position = get_trigger_to_position(sentence, graph)

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
            for u, v, data in graph.out_edges(event, data=True):
                event_type = graph.nodes[event]["type"]
                if v.startswith("E"):
                    raise ValueError("Text graph should only have Event -> Trigger edges")

                if v in trigger_to_span:
                    pos = trigger_to_span[v]
                    if data["type"] == "Trigger":
                        trigger_labels[pos[0]] = self.label_to_id["B-" + event_type]
                        trigger_labels[pos[0]+1: pos[1]] = self.label_to_id["I-" + event_type]
                    else:
                        edge_labels[pos[0]] = self.label_to_id["B-" + data["type"]]
                        edge_labels[pos[0]+1: pos[1]] = self.label_to_id["I-" + data["type"]]

            for mod in graph.nodes[event]["modifications"]:
                mod_labels[self.event_mod_to_id[mod]] = 1

        example["input_ids"] = input_ids
        example["token_type_ids"] = token_type_ids
        example["edge_labels"] = edge_labels
        example["trigger_labels"] = trigger_labels
        example["mod_labels"] = mod_labels
        example["node_type_ids"] = node_types

        id_to_node_type = {v: k for k, v in self.node_type_to_id.items()}

        foo = []
        for tok, nt in zip(self.tokenizer.convert_ids_to_tokens(input_ids.tolist()), node_types):
            foo.append((tok, id_to_node_type[nt.item()]))
        # print(foo[:20])
        # print(foo[-20:])

        return example

    @staticmethod
    def collate_fn(examples):
        keys_to_batch = {"input_ids", "token_type_ids", "attention_mask", "node_type_ids",
                         "mod_labels"}
        batch = defaultdict(list)
        for example in examples:
            for k, v in example.items():
                batch[k].append(v)

        batched_batch  = {}
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

            ann_pred = pred_example["ann"][0]
            sentence = pred_example["sentence"][0]
            encoding_graph = pred_example["encoding_graph"][0]
            graph_pred = filter_graph_to_sentence(ann_pred.text_graph, sentence)
            graph_true = filter_graph_to_sentence(ann_true.text_graph, sentence)
            entity_triggers, event_triggers = self.get_triggers(sentence, graph_true)
            known_events = [n for n in graph_pred.nodes if n.startswith("E")]
            trigger_to_position = get_trigger_to_position(sentence, ann_true)
            for event in self.event_ordering(graph_true):
                event = ann_true.events[event]
                if not ann_pred.contains_event(event):
                    example = self.build_example(ann=ann_true, entity_triggers=entity_triggers,
                                       event=event.id,
                                       event_triggers=event_triggers,
                                       graph=graph_true,
                                       sentence=sentence,
                                       trigger_to_position=trigger_to_position,
                                       encoding_graph=encoding_graph,
                                       known_events=known_events
                                       )
                    if self.example_to_trajectory(example) not in existing_trajectories:
                        n_added += 1
                        if not dry_run:
                            self.examples.append(example)
                        # self.print_example(example)
                    break

        logging.info(f"Added {n_added} new examples from Dagger")

    def example_to_trajectory(self, example):
        tokens = "".join(self.tokenizer.convert_ids_to_tokens(example["input_ids"].tolist(),
                                                      skip_special_tokens=True))
        labels = tuple(example["labels"].tolist())

        return (tokens, labels)


class PC13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.PC13_EVENT_TYPES
    ENTITY_TYPES = consts.PC13_ENTITY_TYPES
    EDGE_TYPES = consts.PC13_EDGE_TYPES
    DUPLICATES_ALLOWED = consts.PC13_DUPLICATES_ALLOWED
    NO_THEME_ALLOWED = consts.PC13_NO_THEME_ALLOWED
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
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Degradation":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False

        if event_type == "Dissociation":
            if arg in {"Cause", "AtLoc", "Site", "ToLoc", "Loc", "FromLoc"}:
                return False


        if "regulation" not in event_type.lower() and reftype not in self.ENTITY_TYPES:
            return False

        return True


def get_free_event_id(graph):
    ids = [int(n[1:]) for n in graph.nodes if n.startswith("E")]
    max_id = max(ids) if ids else 0
    free_id = f"E{max_id+1}"
    return free_id

def get_free_trigger_id(graph):
    ids = [int(n[1:]) for n in graph.nodes if n.startswith("T")]
    max_id = max(ids) if ids else 0
    free_id = f"T{max_id+1}"
    return free_id


def sort_args(args):
    order = ["Theme","Theme2", "Theme3", "Theme4", "Theme5", "Cause", "Site", "CSite", "AtLoc", "ToLoc"]

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
            lines.append(f"{trigger}\t{graph.nodes[trigger]['type']} {' '.join(graph.nodes[trigger]['span'])}\t{graph.nodes[trigger]['text']}")

    for event in [n for n in graph.nodes if n.startswith("E")]:
        try:
            trigger = [v for u, v, d in graph.out_edges(event, data=True) if d["type"] == "Trigger"][0]
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
                args.append(f"{edge_type}{edge_type_count[edge_type]}:{v}") # for Theme2, Product2 etc.
        args = sort_args(args)
        lines.append(f"{event}\t{event_type}:{trigger} {' '.join(args)}")

        for mod in graph.nodes[event]["modifications"]:
            n_mods += 1
            lines.append(f"M{n_mods}\t{mod} {event}")

    return lines

