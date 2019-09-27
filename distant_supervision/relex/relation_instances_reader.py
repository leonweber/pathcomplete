import json
from typing import Set, Tuple, List, Dict, Optional

from dataclasses import dataclass

import logging
import random
from collections import defaultdict
from overrides import overrides

import tqdm

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, MultiLabelField, SequenceLabelField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data import Token

log = logging.getLogger(__name__)  # pylint: disable=invalid-name

NEGATIVE_RELATION_NAME = 'NA'



@DatasetReader.register("relation_instances")
class RelationInstancesReader(DatasetReader):
    r"""DatasetReader to read a relation extraction dataset.

    Each example is a pair of entities, bag (list) of sentences and a relation type. The sentences of each
    bag should be listed consecutively in the dataset file.

    File format: tab separated text file of 7 columns. They are:
        entity1_id
        entity2_id
        entity1_text: can be NA because it is not used by the model
        entity2_text: can be NA because it is not used by the model
        relation_type: use NA to indicate No Relation
        sentence: entity mentions are highlighted with <e1>entity1<\e1> and <e2>entity2<\e2>
        supervision_type: "direct" or "distant"

    The reader assumes that the sentences relevant to a pair of entities are all listed consecutively.
    If the entity pair changes, the reader starts a new bag.

    """

    max_distance = 100  # for position embeddings
    max_sentence_length = None # words

    def __init__(self, lazy: bool = False,
                 max_bag_size: Optional[int] = None,
                 negative_exampels_percentage: int = 100,
                 with_direct_supervision: bool = True,
                 ignore_pairs_without_mentions: bool = True,
                 load_metadata = False,
                 add_inverse=False) -> None:
        """
        args:
            lazy: lazy reading of the dataset
            max_bag_size: maximum number of sentences per a bag
            negative_exampels_percentage: percentage of negative examples to keep
            with_direct_supervision: keep or ignore direct supervision examples
            ignore_pairs_without_mentions: keep or ignore pairs without mentions
        """
        super().__init__(lazy=lazy)
        self.max_bag_size = max_bag_size
        self.negative_exampels_percentage = negative_exampels_percentage
        self.with_direct_supervision = with_direct_supervision
        self.ignore_pairs_without_mentions = ignore_pairs_without_mentions

        self.load_metadata = load_metadata

        self._tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = {"tokens": SingleIdTokenIndexer()}
        self._entity_indexer = {"entities": SingleIdTokenIndexer(namespace="entities")}

        # for logging and input validation
        self._inst_counts: Dict = defaultdict(int)  # count instances per relation type
        self._pairs: Set = set()  # keep track of pairs of entities
        self._bag_sizes: Dict = defaultdict(int)
        self._relation_types_per_bag = defaultdict(int) # count relation types per bag
        self._relation_coocur = defaultdict(int)  # count relation types per bag
        self._relation_coocur: Dict = defaultdict(int)  # count relation types per bag
        self._failed_mentions_count: int = 0  # count mentions with wrong formating
        self._count_direct_supervised_inst: int = 0
        self._count_bag_labels: Dict = defaultdict(int)

        self.add_inverse = False

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            log.info("Reading instances from lines in file at: %s", file_path)

            self._inst_counts = defaultdict(int)  # count instances per relation type
            self._pairs = set()  # keep track of pairs of entities
            self._bag_sizes = defaultdict(int)  
            self._relation_types_per_bag = defaultdict(int) # count relation types per bag
            self._relation_coocur = defaultdict(int)  # count relation types per bag
            self._failed_mentions_count = 0
            self._count_direct_supervised_inst: int = 0
            self._count_bag_labels: Dict = defaultdict(int)

            data = json.load(data_file)

            for pair, pair_data in data.items():
                e1, e2 = pair.split(',')
                rels = pair_data['relations']

                if self.add_inverse:
                    for rel in data[f"{e2},{e1}"]['relations']:
                        rels.append(f"_{rel}_inverse")

                mentions = set(m[0] for m in pair_data['mentions'])
                inst = self.text_to_instance(e1, e2, rels, mentions, is_predict=False, supervision_type='distant')
                if inst is not None:
                    yield inst

            # log relation types and number of instances
            for rel, cnt in sorted(self._inst_counts.items(), key=lambda x: -x[1]):
                log.info("%s - %d", rel, cnt)

            # log number of relations per bag
            log.info("number of relations per bag size (bagsize -> relation count)")
            for k, v in sorted(self._relation_types_per_bag.items(), key=lambda x: -x[1]):
                log.info("%s - %d", k, v)

            for k, v in sorted(self._relation_coocur.items(), key=lambda x: -x[1]):
                log.info("%s - %d", k, v)

    @overrides
    def text_to_instance(self, e1: str, e2: str,  # pylint: disable=arguments-differ
                         rels: Set[str],
                         mentions: Set[str],
                         is_predict: bool,
                         supervision_type: str) -> Instance:
        """Construct an instance given text input.

        is_predict: True if this is being called for prediction not training
        supervision_type: direct or distant

        """
        assert supervision_type in ['direct', 'distant']

        if (e1, e2) in self._pairs and supervision_type == 'distant' and not is_predict:
            assert False, "input file is not sorted, check entities %s, %s" % (e1, e2)
        self._pairs.add((e1, e2))

        for rel in rels:
            self._inst_counts[rel] += 1  # keep track of number of instances in each relation type for logging

        if NEGATIVE_RELATION_NAME in rels:
            if len(rels) > 1:
                log.error("Positive relations between entities can\'t include %s. "
                          "Found relation types: %s between entities %s and %s",
                          NEGATIVE_RELATION_NAME, rels, e1, e2)
            rels.remove(NEGATIVE_RELATION_NAME)

            if random.uniform(0, 1) >= self.negative_exampels_percentage:
                return None

        self._relation_types_per_bag[len(rels)] += 1
        if len(rels) > 1:
            rels_str = ", ".join(sorted(list(rels)))
            self._relation_coocur[rels_str] += 1

        filtered_mentions = list(mentions)[:self.max_bag_size]  # limit number of mentions per bag

        fields_list = []
        for m in filtered_mentions:
            try:
                mention_fields = self._tokens_distances_fields(
                        self._tokenizer.tokenize(m)[:self.max_sentence_length]
                )
                fields_list.append(mention_fields)
            except ValueError as ve:
                # ignore mentions with wrong entity tags
                self._failed_mentions_count += 1
                if self._failed_mentions_count % 1000 == 0:
                    log.error('Number of failed mentions: %d', self._failed_mentions_count)

        if len(fields_list) == 0:
            mention_f = [TextField([Token('.')] * 5, self._token_indexers)]
            position1_f = [SequenceLabelField([-1] * 5, mention_f[0])]
            position2_f = [SequenceLabelField([-1] * 5, mention_f[0])]
        else:
            mention_f, position1_f, position2_f = list(zip(*fields_list))


        if supervision_type == 'direct':
            is_direct_supervision_bag_field = TextField(self._tokenizer.tokenize(". ."), self._token_indexers)
            self._count_direct_supervised_inst += 1
        else:
            is_direct_supervision_bag_field = TextField(self._tokenizer.tokenize("."), self._token_indexers)

        fields = {"entities": TextField([Token(e1), Token(e2)], self._entity_indexer),
                  "mentions": ListField(list(mention_f)),
                  "positions1": ListField(list(position1_f)),
                  "positions2": ListField(list(position2_f)),
                  "is_direct_supervision_bag": is_direct_supervision_bag_field,
                  "has_mentions": LabelField(1 if mentions else 0, skip_indexing=True),
                  "labels": MultiLabelField(rels),  # bag-level labels
                 }
        if self.load_metadata:
            metadata = {
                "mentions": list(mentions),
                "entities": [e1, e2]
            }

            fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def _tokens_distances_fields(self, tokens):
        """Returns the updated list of tokens and entity distances for the first and second entity as fields."""
        tokens, positions1, positions2 = self._tokens_distances(tokens)
        t_f = TextField(tokens, self._token_indexers)
        p1_f = SequenceLabelField(positions1, t_f)
        p2_f = SequenceLabelField(positions2, t_f)
        return t_f, p1_f, p2_f

    def _tokens_distances(self, tokens):
        e1_loc = []
        e2_loc = []

        while len(tokens) < 5:  # a hack to make sure all sentences are at least 5 tokens. CNN breaks otherwise.
            tokens.append(Token(text='.'))

        for i, token in enumerate(tokens):
            if token.text.startswith('<e1>'):
                e1_loc.append((i, 'start'))
                token.text = token.text[4:]
            if token.text.endswith('</e1>'):
                e1_loc.append((i, 'end'))
                token.text = token.text[:-5]
            if token.text.startswith('<e2>'):
                e2_loc.append((i, 'start'))
                token.text = token.text[4:]
            if token.text.endswith('</e2>'):
                e2_loc.append((i, 'end'))
                token.text = token.text[:-5]

        positions1 = self._positions(len(tokens), e1_loc)
        positions2 = self._positions(len(tokens), e2_loc)

        return tokens, positions1, positions2

    def _positions(self, tokens_count: int, e_loc: List[Tuple[int, str]]):
        # if the entity tags are missing, return a list of -1's
        if not e_loc:
            raise ValueError('entity tags are missing.')
        prev_loc = (-10000000000, 'end')  # large negative number
        next_loc_index = 0
        next_loc = e_loc[next_loc_index]
        distance_list = []
        for i in range(tokens_count):
            if prev_loc[1] == 'end' and next_loc[1] == 'start':
                # between two entities
                to_min = [abs(i - prev_loc[0]), abs(i - next_loc[0])]
                to_min.append(self.max_distance)
                distance = min(to_min)
            elif prev_loc[1] == 'start' and next_loc[1] == 'end':
                # inside the same entity
                distance = 0
            else:
                # malformed e_loc
                distance = self.max_distance

            distance_list.append(distance)
            while i == next_loc[0]:
                prev_loc = next_loc
                next_loc_index += 1
                if next_loc_index >= len(e_loc):
                    next_loc = (10000000000, 'start')  # large positive number
                else:
                    next_loc = e_loc[next_loc_index]

        return distance_list