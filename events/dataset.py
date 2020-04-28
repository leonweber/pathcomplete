import itertools
from bisect import bisect_right, bisect_left
from pathlib import Path

import torch_geometric
import spacy
import torch
import transformers
import networkx as nx
from spacy import tokens
from torch_geometric.data import Data
from tqdm import tqdm

from . import consts
from .parse_standoff import StandoffAnnotation


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

        return geometric_graph

    def __getitem__(self, item):
        fname = self.fnames[item]
        result = {"encoding": self.encodings_by_fname[fname],
                  "anntotation": self.annotations_by_fname[fname],
                  "graph": self.geometric_graph_by_fname[fname]
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


class PC13Dataset(BioNLPDataset):
    EVENT_TYPES = consts.PC13_EVENT_TYPES
    ENTITY_TYPES = consts.PC13_ENTITY_TYPES
    EDGE_TYPES = consts.PC_13_EDGE_TYPES

    def __init__(self, path: Path, bert_path: Path):
        super().__init__(path, bert_path, split_sentences=False)
