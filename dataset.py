import logging
import os
import tarfile
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import spacy

from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance, Field, Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from overrides import overrides

import constants

logger = logging.getLogger(__name__)


@DatasetReader.register('streader')
class STReader(DatasetReader):

    def __init__(self, lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.nlp = spacy.load("en_core_sci_sm", disable=['tagger'])

    def get_gogps(self, anns: List[str]) -> Dict[str, Set[str]]:
        gogps: Dict[str, Set[str]] = defaultdict(set)
        for ann in anns:
            ann = ann.split('\t')
            if ann[1].startswith('Gene_or_gene_product'):
                name = ann[2]
                id_ = ann[0]
                gogps[name].add(id_)

        return gogps

    def get_modifications(self, anns: List[str]) -> Dict[str, Set[str]]:
        mods = defaultdict(set)
        for ann in anns:
            if not ann[0].startswith('E'):
                continue

            ann = ann.split('\t')
            type_ = ann[1].split()[0].split(':')[0]

            if type_ in constants.SIMPLE_EVENTS:
                theme_ids = [i.split(':')[1] for i in ann[1].split() if i.startswith('Theme')]
                assert len(theme_ids) == 1
                theme_id = theme_ids[0]

                mods[theme_id].add(type_)

        return mods

    @overrides
    def _read(self, file_path: str):
        logger.info(f"Reading data from {file_path}")
        with tarfile.open(cached_path(file_path), 'r') as data_file:
            file_ids = set(os.path.splitext(n)[0] for n in data_file.getnames() if 'PMID' in n)
            for file_id in list(file_ids)[:2]:
                with data_file.extractfile(file_id + '.txt') as f:
                    text = f.read().decode()
                with data_file.extractfile(file_id + '.a1') as f:
                    a1 = f.read().decode().splitlines()
                with data_file.extractfile(file_id + '.a2') as f:
                    a2 = f.read().decode().splitlines()

                gogps = self.get_gogps(a1 + a2)
                mods = self.get_modifications(a1 + a2)

                for gogp, ids in gogps.items():
                    true_mods = set()
                    for id_ in ids:
                        true_mods.update(mods[id_])

                    false_mods = set(constants.SIMPLE_EVENTS) - true_mods

                    for mod in true_mods:
                        yield self.text_to_instance(text, gogp, mod, 1)
                    for mod in false_mods:
                        yield self.text_to_instance(text, gogp, mod, 0)

    @overrides
    def text_to_instance(self, text: str, gene: str, mod: str, label: int) -> Instance:
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)[:250] + [Token('[SEP]')] + self._tokenizer.tokenize(mod + ' ' + gene)
        fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        fields['label'] = LabelField(label, skip_indexing=True)

        return Instance(fields)

