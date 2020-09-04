import itertools
import math
import re
from bisect import bisect_right, bisect_left
from collections import defaultdict, deque
from operator import attrgetter, itemgetter
from pathlib import Path

import networkx as nx
import transformers
import torch
from flair.data import Span, Sentence
from flair.tokenization import SciSpacySentenceSplitter
from tqdm import tqdm
import stanza

from events import consts
from events.parse_standoff import StandoffAnnotation



def get_triggers(sent, ann: StandoffAnnotation):
    entity_trigger = []
    event_trigger = []
    for trigger_id, trigger in ann.triggers.items():
        if sent.end_pos > int(trigger.start) >= sent.start_pos:
            if trigger in ann.entity_triggers:
                entity_trigger.append(trigger)
            else:
                event_trigger.append(trigger)

    return entity_trigger, event_trigger


def get_trigger_to_position(sent, ann: StandoffAnnotation):
    trigger_to_position = {}
    for trigger_id, trigger in ann.triggers.items():
        if sent.end_pos > int(trigger.start) >= sent.start_pos:
            trigger_to_position[trigger_id] = (int(trigger.start) - sent.start_pos,
                                               int(trigger.end) - sent.start_pos)
    return trigger_to_position




def adapt_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """
    start = int(start)
    end = int(end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return new_start, new_end


def get_trigger_spans(triggers, sentence: Sentence, add_labels=False):
    trigger_spans = {}

    token_starts = [i.start_pos + sentence.start_pos for i in sentence.tokens]
    for trigger in triggers:
        token_start, token_end = adapt_span(start=trigger.start, end=trigger.end, token_starts=token_starts)
        # assert sentence.to_original_text()[sentence.tokens[token_start].start_pos:sentence.tokens[token_end-1].end_pos] == trigger.text
        trigger_spans[trigger.id] = (token_start, token_end)

        if add_labels:
            sentence.tokens[token_start].remove_labels("trigger")
            sentence.tokens[token_start].add_label("trigger", "B-Trigger")
            for i in range(token_start+1, token_end):
                sentence.tokens[i].remove_labels("trigger")
                sentence.tokens[i].add_label("trigger", "I-Trigger")

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


def get_partial_graph(ann, known_events, triggers, trigger_spans):
    graph = ann.event_graph.copy()
    for event in known_events:
        for u, v, d in ann.event_graph.out_edges(event, data=True):
            if v.startswith("E") and v not in known_events:
                new_v = ann.events[v].trigger.id
                graph.add_edge(u, new_v, **d)
    triggers = [i.id for i in triggers]
    graph = graph.subgraph(triggers + list(known_events))

    node_to_span = {}
    for node in graph.nodes:
        if node in trigger_spans:
            node_to_span[node] = {"span": trigger_spans[node]}
        else:
            node_to_span[node] = {"span": trigger_spans[ann.events[node].trigger.id]}
    nx.set_node_attributes(graph, node_to_span)


    return graph




class BioNLPDataset:
    EVENT_TYPES = None
    ENTITY_TYPES = None
    EDGE_TYPES = None
    DUPLICATES_ALLOWED = None
    NO_THEME_ALLOWED = None
    EVENT_TYPE_TO_ORDER = None
    EDGE_TYPES_TO_MOD = None


    def is_valid_argument_type(self, arg, reftype):
        raise NotImplementedError


    @staticmethod
    def collate(batch):
        assert len(batch) == 1
        return batch[0]

    def __init__(self, path: Path, tokenizer: Path,
                 linearize_events: bool = False, batch_size: int = 16, predict: bool = False,
                 trigger_ordering: str = "position", predict_entities: bool = False,
                 max_span_width: int = 10, trigger_detector = None, small: bool = False
                 ):
        self.trigger_detector = trigger_detector
        if small:
            self.text_files = [f for f in path.glob('*.txt')][:5]
        else:
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
        self.predict_entities = predict_entities
        self.max_span_width = max_span_width
        # self.stanza = stanza.Pipeline("en", package="craft")

        if trigger_ordering  == "id":
            self.trigger_ordering = lambda x, y: x
        elif trigger_ordering == "position":
            self.trigger_ordering = self.sort_triggers_by_position
        elif trigger_ordering == "simple_first":
            self.trigger_ordering = self.sort_triggers_by_simple_first

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(str(tokenizer))
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["@"]})

        self.examples = []
        self.predicted_examples = []
        self.predict_example_by_fname = {}
        self.trigger_detection_examples = []

        for file in tqdm(self.text_files, desc="Parsing raw data"):
            if file.with_suffix(".a2").exists():
                with file.with_suffix(".a2").open() as f:
                    a2_lines = f.readlines()
            else:
                a2_lines = []
            with file.open() as f, file.with_suffix(".a1").open() as f_a1:
                text = f.read()
                sentences = self.sentence_splitter.split(text)
                a1_lines = f_a1.readlines()
                ann = StandoffAnnotation(a1_lines=a1_lines, a2_lines=a2_lines)
                # if isinstance(self.trigger_detector, dict):
                #     predicted_a2_trigger_lines = self.trigger_detector[file.name]
                # else:
                #     predicted_a2_trigger_lines = get_event_trigger_lines_from_sentences(sentences, self.trigger_detector, len(a1_lines))
                # ann_predicted = StandoffAnnotation(a1_lines=[], a2_lines=predicted_a2_trigger_lines)
                # ann_predicted = integrate_predicted_a2_triggers(ann, ann_predicted)
                new_examples =  self.generate_examples(text, ann)
                for example in new_examples:
                    example["fname"] = file.name
                self.examples += new_examples
                # self.examples += self.generate_examples(text, ann_predicted)
                # self.trigger_detection_examples += self.generate_trigger_detection_examples(sentences, ann)
                self.predict_example_by_fname[file.name] = (file.name, text, ann)

        self.fnames = sorted(self.predict_example_by_fname)



    def sort_triggers_by_position(self, triggers, ann):
        return sorted(triggers, key=lambda x: int(x.start))

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
            # implement multitask data generation here, hacky but should work :)
            if item >= len(self):
                raise IndexError
            # trigger_detection_example = self.trigger_detection_examples[item % len(self.trigger_detection_examples)]
            event_generation_example = self.examples[item % len(self.examples)]

            example = {}
            # for k, v in trigger_detection_example.items():
            #     example["td_" + k] = v
            for k, v in event_generation_example.items():
                example["eg_" + k] = v

            return example

    def __len__(self):
        if self.predict:
            return len(self.fnames)
        else:
            return max(len(self.examples), len(self.trigger_detection_examples))

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
            known_events = set()
            entity_triggers, event_triggers = get_triggers(sentence, ann)
            for token in sentence.tokens:
                token.add_label("trigger", "O")
            trigger_spans = get_trigger_spans(entity_triggers, sentence)
            trigger_spans.update(get_trigger_spans(event_triggers, sentence,
                                                   add_labels=True))
            for i_trigger, trigger in enumerate(self.trigger_ordering(event_triggers, ann)):
                for event in trigger_to_events[trigger.id]:
                    example = self.build_example(ann=ann,
                                                 entity_triggers=entity_triggers,
                                                 event=event,
                                                 event_triggers=event_triggers,
                                                 known_events=known_events,
                                                 sentence=sentence,
                                                 trigger_spans=trigger_spans,
                                                 trigger=trigger)
                    # if len(example["graph"].nodes) > 1:
                    examples.append(example)
                    known_events.add(event.id)
                example = self.build_example(ann=ann,
                                             entity_triggers=entity_triggers,
                                             event=None,
                                             event_triggers=event_triggers,
                                             known_events=known_events,
                                             sentence=sentence,
                                             trigger_spans=trigger_spans,
                                             trigger=trigger)
                examples.append(example)

        return examples

    def trigger_ordering(self, triggers):
        return triggers

    def build_example(self, ann, entity_triggers, event,
                      event_triggers, known_events, sentence, trigger_spans,
                      trigger):
        edges_to_predict = {}
        if event:
            for u, v, d in ann.text_graph.out_edges(event.id, data=True):
                edges_to_predict[(event.id, v)] = d["type"]
            event_id = event.id
        else:
            event_id = "None"

        example = {"sentence": sentence,
                   "trigger": trigger.id, "event_id": event_id,
                   "graph": get_partial_graph(ann, known_events,
                                              triggers=entity_triggers + event_triggers,
                                              trigger_spans = trigger_spans),
                   "edges_to_predict": edges_to_predict,
                   }

        return example

    @staticmethod
    def collate_fn(examples):
        keys_to_batch = {"input_ids", "token_type_ids", "attention_mask", "span_mask", "td_labels"}
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



class PC13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.PC13_EVENT_TYPES
    ENTITY_TYPES = consts.PC13_ENTITY_TYPES
    EDGE_TYPES = consts.PC13_EDGE_TYPES
    DUPLICATES_ALLOWED = consts.PC13_DUPLICATES_ALLOWED
    NO_THEME_ALLOWED = consts.PC13_NO_THEME_ALLOWED
    EVENT_TYPE_TO_ORDER = consts.PC13_EVENT_TYPE_TO_ORDER
    EDGE_TYPES_TO_MOD = consts.PC13_EDGE_TYPES_TO_MOD

    def __init__(self, path: Path, bert_path: Path,
                 batch_size: int = 16, predict=False, trigger_ordering="position",
                 linearize_events: bool = False, trigger_detector=None, small=False):
        super().__init__(path, bert_path, batch_size=batch_size, predict=predict,
                         linearize_events=linearize_events,
                         trigger_ordering=trigger_ordering,
                         trigger_detector=trigger_detector,
                         small=small
                         )

    def is_valid_argument_type(self, event_type, arg, reftype, refid):
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


        if arg == "Cause" or re.match(r'^(Theme|Product)\d*$', arg):
            return reftype in self.ENTITY_TYPES or reftype in self.EVENT_TYPES
        elif arg in ("ToLoc", "AtLoc", "FromLoc"):
            return reftype in ("Cellular_component", )
        elif re.match(r'^C?Site\d*$', arg):
            return reftype in ("Simple_chemical", )
        elif re.match(r'^Participant\d*$', arg):
            return reftype in self.ENTITY_TYPES
        else:
            return False


class GE13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.GE_EVENT_TYPES
    ENTITY_TYPES = consts.GE_ENTITY_TYPES
    EDGE_TYPES = consts.GE_EDGE_TYPES
    DUPLICATES_ALLOWED = consts.GE_DUPLICATES_ALLOWED
    NO_THEME_FORBIDDEN = consts.GE_NO_THEME_FORBIDDEN
    EVENT_TYPE_TO_ORDER = consts.PC13_EVENT_TYPE_TO_ORDER

    def __init__(self, path: Path, bert_path: Path,
                 batch_size: int = 16, predict=False, trigger_ordering="position",
                 linearize_events: bool = False, trigger_detector=None, small=False):
        super().__init__(path, bert_path, batch_size=batch_size, predict=predict,
                         linearize_events=linearize_events,
                         trigger_ordering=trigger_ordering,
                         trigger_detector=trigger_detector,
                         small=small
                         )

    def is_valid_argument_type(self, arg, reftype):
        if arg == "Cause" or re.match(r'^(Theme|Product)\d*$', arg):
            return reftype in self.ENTITY_TYPES or reftype in self.EVENT_TYPES
        elif arg in ("ToLoc", "AtLoc", "FromLoc"):
            return reftype in ("Cellular_component", )
        elif re.match(r'^C?Site\d*$', arg):
            return reftype in ("Simple_chemical", )
        elif re.match(r'^Participant\d*$', arg):
            return reftype in self.ENTITY_TYPES
        else:
            return False

def get_event_trigger_lines_from_sentences(sentences, n_a1_lines):
    lines = []
    for sentence in sentences:
        for trigger in sentence.get_spans("trigger"):
            start = trigger.start_pos + sentence.start_pos
            end = trigger.end_pos + sentence.start_pos
            id_ = f"T{len(lines) + n_a1_lines + 1}"
            lines.append(f"{id_}\tNone {start} {end}\t{sentence.to_original_text()[trigger.start_pos:trigger.end_pos]}")

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


def get_a2_lines_from_graph(graph: nx.DiGraph, event_types):
    lines = []

    for trigger, d in graph.nodes(data=True):
        if trigger.startswith("T") and d["type"] in event_types:
            lines.append(f"{trigger}\t{graph.nodes[trigger]['type']} {' '.join(graph.nodes[trigger]['orig_span'])}\t{graph.nodes[trigger]['text']}")

    for event in [n for n in graph.nodes if n.startswith("E")]:
        trigger = [u for u, v, d in graph.in_edges(event, data=True) if d["type"] == "Trigger"][0]
        event_type = graph.nodes[event]["type"]
        args = []
        edge_type_count = defaultdict(int)
        for _, v, d in graph.out_edges(event, data=True):
            edge_type = d['type']
            edge_type_count[edge_type] += 1
            if edge_type_count[edge_type] == 1:
                args.append(f"{edge_type}:{v}")
            else:
                args.append(f"{edge_type}{edge_type_count[edge_type]}:{v}") # for Theme2, Product2 etc.
            pass
        lines.append(f"{event}\t{event_type}:{trigger} {' '.join(args)}")

    return lines