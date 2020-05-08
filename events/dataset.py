import itertools
from bisect import bisect_right, bisect_left
from pathlib import Path
import random

import torch_geometric
import spacy
import torch
import transformers
import networkx as nx
from spacy import tokens
from torch_geometric.data import Data
from tqdm import tqdm

from events import consts
from events.parse_standoff import StandoffAnnotation


def adapt_span(start, end, token_starts):
    """
    Adapt annotations to token spans
    """
    start = int(start)
    end = int(end)

    new_start = bisect_right(token_starts, start) - 1
    new_end = bisect_left(token_starts, end)

    return new_start, new_end



class BioNLPDataset:
    EVENT_TYPES = None
    ENTITY_TYPES = None
    EDGE_TYPES = None

    @staticmethod
    def collate(batch):
        assert len(batch) == 1
        return batch[0]

    def __init__(self, path: Path, tokenizer: Path, split_sentences: bool = False):
        self.max_len = 512
        self.split_sentences = split_sentences
        self.text_files = [f for f in path.glob('*.txt')]
        self.node_type_to_id = {v: i for i, v in enumerate(sorted(
            itertools.chain(self.EVENT_TYPES, self.ENTITY_TYPES)))}
        self.edge_type_to_id = {v: i for i, v in enumerate(sorted(self.EDGE_TYPES))}
        self.nlp = spacy.load("en_core_sci_sm", disable=['tagger'])

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(str(tokenizer))

        self.encodings_by_fname = {}
        self.annotations_by_fname = {}
        self.geometric_graph_by_fname = {}
        for file in tqdm(self.text_files, desc="Parsing raw data"):
            with file.open() as f:
                encoding = self.tokenizer.tokenizer.encode(f.read())
                self.add_sentence_ids(encoding)
                self.encodings_by_fname[file.name] = encoding

                ann = StandoffAnnotation(file.with_suffix(".a1"), file.with_suffix(".a2"))
                self.annotations_by_fname[file.name] = ann
                self.geometric_graph_by_fname[file.name] = self.nx_to_geometric(ann.event_graph)
                self.add_annotation_information(graph=self.geometric_graph_by_fname[file.name],
                                                ann=ann,
                                                encoding=encoding)

        self.fnames = sorted(self.encodings_by_fname)

    def add_sentence_ids(self, encoding):
        doc: spacy.tokens.Doc = self.nlp(str(encoding.original_str))
        sentence_ids = []
        sentence_ends = [s.end_char for s in doc.sents]
        i_sent = 0
        for tok_start, tok_end in encoding.offsets:
            if tok_start > sentence_ends[i_sent]:
                i_sent += 1
            sentence_ids.append(i_sent)
        encoding.sentence_ids = sentence_ids

    def nx_to_geometric(self, graph):
        geometric_graph = torch_geometric.utils.from_networkx(graph)
        geometric_graph.node_names = list(graph.nodes)
        geometric_graph.name_to_id = {n: i for i, n in enumerate(geometric_graph.node_names)}
        geometric_graph.id_to_name = {i: n for n, i in geometric_graph.name_to_id.items()}
        geometric_graph.topological_order = [geometric_graph.name_to_id[i] for i in nx.topological_sort(graph) if 'T' not in i]
        geometric_graph.node_type = [self.node_type_to_id[n['type']] for _, n in graph.nodes(data=True)]
        geometric_graph.edge_attr = torch.tensor([self.edge_type_to_id[e['type']] for _, _, e in graph.edges(data=True)])
        geometric_graph.entities = [geometric_graph.name_to_id[n] for n in geometric_graph.node_names if 'T' in n]
        geometric_graph.G = graph

        return geometric_graph

    def __getitem__(self, item):
        fname = self.fnames[item]
        g = self.geometric_graph_by_fname[fname]
        order = self.sample_order(g)
        node_targets = self.compute_targets(g, order)
        result = {"encoding": self.encodings_by_fname[fname],
                  "graph": g,
                  "node_targets": node_targets,
                  "order": order,
                  "fname": fname
                  }

        return result

    def __len__(self):
        return len(self.fnames)

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

    def add_annotation_information(self, graph, ann, encoding):
        node_spans = []
        trigger_to_span = {}
        event_to_trigger = {}
        token_starts = [i[1] for i in encoding.offsets[1:-1]] # because of CLS and SEP tokens
        n_tokens = len(encoding)

        for trigger in ann.triggers.values():
            trigger_to_span[trigger.id] = adapt_span(start=trigger.start,
                                                    end=trigger.end,
                                                    token_starts=token_starts)
        for event in ann.events.values():
            try:
                event_to_trigger[event.id] = event.trigger.id
            except AttributeError:
                event_to_trigger[event.id] = event.trigger

        for node in graph.node_names:
            if node in event_to_trigger:
                trigger = event_to_trigger[node]
            else:
                trigger = node

            if trigger in trigger_to_span:
                span = trigger_to_span[trigger]
            else:
                span = (n_tokens-2, n_tokens-1) # default to SEP for failed parses

            span = (span[0]+1, span[1]+1) # because of CLS token
            node_spans.append(span)

        graph.node_spans = node_spans

    def sample_order(self, g):
        order = []
        G = g.G
        indegree_map = {v: d for v, d in G.in_degree() if d > 0 if "T" not in v}
        # These nodes have zero indegree and ready to be returned.
        zero_indegree = [v for v, d in G.in_degree() if d == 0 if "T" not in v]

        while zero_indegree:
            node = random.sample(zero_indegree, 1)[0]
            zero_indegree.remove(node)
            for _, child in G.edges(node):
                if "T" in child:
                    continue
                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]
            order.append(g.name_to_id[node])

        return order

    def compute_targets(self, g, order):
        targets = torch.zeros((len(order) + 1, len(self.node_type_to_id)))
        G = nx.convert_node_labels_to_integers(g.G).subgraph(order)
        indegree_map = {v: d for v, d in G.in_degree() if d > 0}
        zero_indegree = [v for v, d in G.in_degree() if d == 0]

        for valid_node in zero_indegree:
            targets[0, g.node_type[valid_node]] += 1
        for i, predicted_node in enumerate(order[:-1], start=1):
            zero_indegree.remove(predicted_node)
            for _, child in G.edges(predicted_node):
                indegree_map[child] -= 1
                if indegree_map[child] == 0:
                    zero_indegree.append(child)
                    del indegree_map[child]

            for valid_node in zero_indegree:
                targets[i, g.node_type[valid_node]] += 1

        targets[-1][self.node_type_to_id["None"]] = 1
        targets = targets / targets.sum(dim=1).unsqueeze(1)

        return targets



class PC13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.PC13_EVENT_TYPES
    ENTITY_TYPES = consts.PC13_ENTITY_TYPES
    EDGE_TYPES = consts.PC_13_EDGE_TYPES

    def __init__(self, path: Path, bert_path: Path):
        super().__init__(path, bert_path, split_sentences=False)
