import itertools
from bisect import bisect_right, bisect_left
from collections import defaultdict
from copy import deepcopy
from operator import attrgetter
from pathlib import Path
import random

import networkx as nx
import spacy
import transformers
import torch
from flair.models import SequenceTagger
from flair.tokenization import SciSpacySentenceSplitter
from tqdm import tqdm

from events import consts
from events.parse_standoff import StandoffAnnotation


MAX_LEN = 256


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

def get_event_linearization(ann, tokenizer, node_type_to_id, known_events=None):
    text = ""
    node_char_spans = []
    node_types = []
    node_spans = []
    node_ids = []

    if known_events is not None:
        events = [ann.events[e] for e in known_events]
    else:
        events = ann.events.values()
    events = sorted(events, key=attrgetter("trigger.start"))

    for event in events:
        start = len(text) + 1
        node_char_spans.append((start, start+len(event.trigger.text)))
        node_types.append(event.trigger.type)
        node_ids.append(event.id)
        text += " " + event.trigger.text

        for u, v, data in ann.event_graph.out_edges(event.id, data=True):
            try:
                trigger = ann.triggers[v]
            except KeyError:
                trigger = ann.events[v].trigger
            start = len(text) + 1
            node_char_spans.append((start, start+len(trigger.text)))
            text += " " + trigger.text
            node_ids.append(v)
            node_types.append(trigger.type)
        text += " |"
    text += "[SEP]"

    encoding = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
    token_starts = [i[0] for i in encoding["offset_mapping"]][:-1]
    node_type_tensor = torch.zeros(len(encoding["input_ids"]))
    node_type_tensor[:] = node_type_to_id["None"]
    for (start, end), node_type in zip(node_char_spans, node_types):
        start, end = adapt_span(start=start, end=end, token_starts=token_starts)
        node_type_tensor[start:end] = node_type_to_id[node_type]
        node_spans.append((start,end))

    return encoding, node_type_tensor, node_spans, node_ids


def get_triggers(sent, ann: StandoffAnnotation):
    entity_trigger_ids = []
    event_trigger_ids = []
    for trigger_id, trigger in ann.triggers.items():
        if sent.end_pos > int(trigger.start) >= sent.start_pos:
            if trigger in ann.entity_triggers:
                entity_trigger_ids.append(trigger_id)
            else:
                event_trigger_ids.append(trigger_id)

    return entity_trigger_ids, event_trigger_ids


def get_trigger_to_position(sent, ann: StandoffAnnotation):
    trigger_to_position = {}
    for trigger_id, trigger in ann.triggers.items():
        if sent.end_pos > int(trigger.start) >= sent.start_pos:
            trigger_to_position[trigger_id] = (int(trigger.start) - sent.start_pos,
                                               int(trigger.end) - sent.start_pos)
    return trigger_to_position


def get_text_encoding_and_node_spans(text, trigger_pos, tokenizer, max_length, nodes,
                                     trigger_to_position, node_type_to_id, ann):
    marker_start, marker_end = trigger_pos
    marked_text = text[:marker_start] + "@ " + text[ marker_start:marker_end] + " @" + text[
                                                                           marker_end:]
    encoding_text = tokenizer.encode_plus(marked_text, return_offsets_mapping=True, max_length=max_length, add_special_tokens=True,
                                               return_overflowing_tokens=True, pad_to_max_length=True)
    token_starts = [i for i, _
                     in tokenizer.encode_plus(text, return_offsets_mapping=True,
                                              add_special_tokens=True)["offset_mapping"][1:-1]]
    node_spans = get_trigger_spans(triggers=nodes, token_starts=token_starts,
                                   marker_start=marker_start, marker_end=marker_end,
                                   trigger_to_position=trigger_to_position, text=text,
                                   encoding_text=encoding_text, tokenizer=tokenizer)
    node_types_text = torch.zeros(len(encoding_text["input_ids"]))
    node_types_text[:] = node_type_to_id["None"]
    for node, span in zip(nodes, node_spans):
        node = ann.triggers[node]
        node_types_text[span[0]:span[1]] = node_type_to_id[node.type]


    return encoding_text, node_spans, node_types_text




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
        if char_start >= marker_start:
            token_start += 1
        if char_start >= marker_end:
            token_start += 1

        if char_end > marker_start:
            token_end += 1
        if char_end > marker_end:
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




class BioNLPDataset:
    EVENT_TYPES = None
    ENTITY_TYPES = None
    EDGE_TYPES = None
    DUPLICATES_ALLOWED = None
    NO_THEME_FORBIDDEN = None
    EDGES_FORBIDDEN = None


    @staticmethod
    def collate(batch):
        assert len(batch) == 1
        return batch[0]

    def __init__(self, path: Path, tokenizer: Path, trigger_detector: SequenceTagger,
                 linearize_events: bool = False, batch_size: int = 16, predict: bool = False):
        self.text_files = [f for f in path.glob('*.txt')]
        self.node_type_to_id = {}
        for i in sorted(itertools.chain(self.EVENT_TYPES, self.ENTITY_TYPES)):
            if i not in self.node_type_to_id:
                self.node_type_to_id[i] = len(self.node_type_to_id)

        self.edge_type_to_id = {v: i for i, v in enumerate(sorted(self.EDGE_TYPES))}
        self.sentence_splitter = SciSpacySentenceSplitter()
        self.batch_size = batch_size
        self.predict = predict
        self.linearize_events = linearize_events
        self.trigger_detector = trigger_detector

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(str(tokenizer))
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["@"]})

        self.examples = []
        self.predicted_examples = []
        self.predict_example_by_fname = {}
        for file in tqdm(self.text_files, desc="Parsing raw data"):
            with file.open() as f, file.with_suffix(".a1").open() as f_a1, file.with_suffix(".a2").open() as f_a2:
                text = f.read()
                sentences = self.sentence_splitter.split(text)
                a1_lines = f_a1.readlines()
                a2_lines = f_a2.readlines()
                ann = StandoffAnnotation(a1_lines=a1_lines, a2_lines=a2_lines)
                predicted_a2_trigger_lines = get_event_trigger_lines_from_sentences(sentences, self.trigger_detector, len(a1_lines))
                ann_predicted = StandoffAnnotation(a1_lines=[], a2_lines=predicted_a2_trigger_lines)
                ann_predicted = integrate_predicted_a2_triggers(ann, ann_predicted)
                self.examples += self.generate_examples(text, ann)
                self.predicted_examples += self.generate_examples(text, ann_predicted)
                self.predict_example_by_fname[file.name] = (file.name, text, ann)

        self.fnames = sorted(self.predict_example_by_fname)

    def __getitem__(self, item):

        if self.predict:
            fname = self.fnames[item]
            return self.predict_example_by_fname[fname]
        else:
            return self.predicted_examples[item]

    def __len__(self):
        if self.predict:
            return len(self.fnames)
        else:
            return len(self.predicted_examples)

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
            entity_triggers, event_triggers = get_triggers(sentence, ann)
            trigger_to_position = get_trigger_to_position(sentence, ann)

            known_events = []

            for i_trigger, trigger_id in enumerate(event_triggers):
                for event in trigger_to_events[trigger_id]:
                    example = self.build_example(ann, entity_triggers, event,
                                                 event_to_trigger, event_triggers,
                                                 i_trigger, known_events, sentence,
                                                 trigger_id, trigger_to_position)

                    known_events.append(event.id)
                    examples.append(example)

                # signal end of trigger
                example = self.build_example(ann, entity_triggers, None,
                                             event_to_trigger, event_triggers,
                                             i_trigger, known_events, sentence,
                                             trigger_id, trigger_to_position)
                examples.append(example)

        return examples

    def build_example(self, ann, entity_triggers, event, event_to_trigger,
                      event_triggers, i_trigger, known_events, sentence, trigger_id,
                      trigger_to_position):
        example = {}
        if self.linearize_events:
            encoding_graph, node_types_graph, node_spans_graph, node_ids_graph = get_event_linearization(
                ann, known_events=known_events,
                tokenizer=self.tokenizer, node_type_to_id=self.node_type_to_id)
        else:
            encoding_graph, node_types_graph, node_spans_graph, node_ids_graph = get_event_graph(
                known_events=known_events,
                ann=ann, tokenizer=self.tokenizer,
                node_type_to_id=self.node_type_to_id,
                entities=entity_triggers)
        adjacency_matrix = get_adjacency_matrix(event_graph=ann.event_graph,
                                                nodes_text=entity_triggers + event_triggers,
                                                nodes_graph=node_ids_graph,
                                                event_to_trigger=event_to_trigger,
                                                edge_type_to_id=self.edge_type_to_id)
        assert adjacency_matrix["text_to_graph"].shape[1] == len(node_spans_graph)
        remaining_length = MAX_LEN - len(encoding_graph["input_ids"])
        node_spans_graph = [(a + remaining_length, b + remaining_length) for a, b in
                            node_spans_graph]  # because we will append them to the text encoding
        encoding_text, node_spans_text, node_types_text = get_text_encoding_and_node_spans(
            text=sentence.to_original_text(),
            trigger_pos=trigger_to_position[trigger_id],
            tokenizer=self.tokenizer,
            max_length=remaining_length,
            nodes=entity_triggers + event_triggers,
            node_type_to_id=self.node_type_to_id,
            ann=ann,
            trigger_to_position=trigger_to_position)
        input_ids = torch.cat([torch.tensor(encoding_text["input_ids"]),
                               torch.tensor(encoding_graph["input_ids"])])
        token_type_ids = torch.zeros(input_ids.size(0))
        token_type_ids[len(encoding_text["input_ids"]):] = 1
        edge_types = {}

        if event:
            for u, v, data in ann.event_graph.out_edges(event.id, data=True):
                if v.startswith("E"):
                    try:
                        v = ann.events[v].trigger.id
                    except AttributeError:
                        v = ann.events[v].trigger

                edge_types[(u, v)] = data["type"]
            labels = []
            for node in entity_triggers + event_triggers:
                node = ann.triggers[node]
                edge_type = edge_types.get((event.id, node.id), "None")
                labels.append(self.edge_type_to_id[edge_type])
        else:
            labels = torch.tensor([self.edge_type_to_id["None"]] * len(entity_triggers + event_triggers))

        example["input_ids"] = input_ids
        example["token_type_ids"] = token_type_ids
        example["adjacency_matrix"] = adjacency_matrix
        example["node_types_text"] = node_types_text
        example["node_types_graph"] = node_types_graph
        example["node_spans_text"] = node_spans_text
        example["node_spans_graph"] = node_spans_graph
        example["trigger_span"] = node_spans_text[len(entity_triggers) + i_trigger]
        example["labels"] = labels
        return example

    @staticmethod
    def collate_fn(examples):
        keys_to_batch = {"input_ids", "token_type_ids"}
        batch = defaultdict(list)
        for example in examples:
            for k, v in example.items():
                batch[k].append(v)
        for k in keys_to_batch:
            batch[k] = torch.stack(batch[k])

        return batch



class PC13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.PC13_EVENT_TYPES
    ENTITY_TYPES = consts.PC13_ENTITY_TYPES
    EDGE_TYPES = consts.PC13_EDGE_TYPES
    DUPLICATES_ALLOWED = consts.PC13_DUPLICATES_ALLOWED
    NO_THEME_FORBIDDEN = consts.PC13_NO_THEME_FORBIDDEN
    EDGES_FORBIDDEN = consts.PC13_EDGES_FORBIDDEN

    def __init__(self, path: Path, bert_path: Path,trigger_detector: SequenceTagger,
                 batch_size: int = 16, predict=False,
                 linearize_events: bool = False, ):
        super().__init__(path, bert_path, batch_size=batch_size, predict=predict,
                         linearize_events=linearize_events, trigger_detector=trigger_detector)


def get_event_trigger_lines_from_sentences(sentences, trigger_detector, n_a1_lines):
    lines = []
    trigger_detector.predict(sentences)
    for sentence in sentences:
        for trigger in sentence.get_spans("trigger"):
            start = trigger.start_pos + sentence.start_pos
            end = trigger.end_pos + sentence.start_pos
            id_ = f"T{len(lines) + n_a1_lines + 1}"
            lines.append(f"{id_}\t{trigger.tag} {start} {end}\t{trigger.text}")

    return lines


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


def get_a2_lines_from_graph(graph: nx.DiGraph):
    lines = []
    for event in [n for n in graph.nodes if n.startswith("E")]:
        trigger = [u for u, v, d in graph.in_edges(event, data=True) if d["type"] == "Trigger"][0]
        event_type = graph.nodes[event]["type"]
        args = []
        for _, v, d in graph.out_edges(event, data=True):
            args.append(f"{d['type']}:{v}")
        lines.append(f"{event}\t{event_type}:{trigger} {' '.join(args)}")

    return lines