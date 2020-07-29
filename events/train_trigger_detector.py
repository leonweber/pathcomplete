import os

import argparse
from pathlib import Path
from typing import Union, Callable, Tuple, List

import flair
from flair.datasets import biomedical, ColumnCorpus
from flair.datasets.biomedical import CoNLLWriter, InternalBioNerDataset, Entity
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, \
    FlairEmbeddings, PooledFlairEmbeddings, CharacterEmbeddings, FastTextEmbeddings
from flair.models import SequenceTagger
from flair.tokenization import SciSpacySentenceSplitter
from flair.trainers import ModelTrainer

class BIONLP2013_PC_TRIGGERS(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            tokenizer: Callable[[str], Tuple[List[str], List[int]]] = None,
            sentence_splitter: Callable[[str], Tuple[List[str], List[int]]] = None,
    ):
        """
           :param base_path: Path to the corpus on your machine
           :param in_memory: If True, keeps dataset in memory giving speedups in training.
           :param tokenizer: Callable that segments a sentence into words,
                             defaults to scispacy
           :param sentence_splitter: Callable that segments a document into sentences,
                                     defaults to scispacy
           """


        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "trigger", 2: "space-after"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"

        if not (train_file.exists() and dev_file.exists()):
            train_folder, dev_folder, test_folder = biomedical.BIONLP2013_PC.download_corpus(data_folder / "original")
            train_data = self.parse_input_files(train_folder)
            dev_data = self.parse_input_files(dev_folder)

            sentence_splitter = SciSpacySentenceSplitter()

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)

        super(BIONLP2013_PC_TRIGGERS, self).__init__(
            data_folder, columns, tag_to_bioes="trigger", in_memory=in_memory
        )


    @staticmethod
    def parse_input_files(input_folder: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        for txt_file in input_folder.glob("*.txt"):
            name = txt_file.with_suffix("").name
            a1_file = txt_file.with_suffix(".a2")

            with txt_file.open() as f:
                documents[name] = f.read()

            with a1_file.open() as ann_reader:
                entities = []

                for line in ann_reader:
                    fields = line.strip().split("\t")
                    if fields[0].startswith("T"):
                        ann_type, start, end = fields[1].split()
                        entities.append(
                            Entity(
                                char_span=(int(start), int(end)), entity_type=ann_type
                            )
                        )
                entities_per_document[name] = entities

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )

class BIONLP2013_CG_TRIGGERS(ColumnCorpus):
    def __init__(
            self,
            base_path: Union[str, Path] = None,
            in_memory: bool = True,
            tokenizer: Callable[[str], Tuple[List[str], List[int]]] = None,
            sentence_splitter: Callable[[str], Tuple[List[str], List[int]]] = None,
    ):
        """
           :param base_path: Path to the corpus on your machine
           :param in_memory: If True, keeps dataset in memory giving speedups in training.
           :param tokenizer: Callable that segments a sentence into words,
                             defaults to scispacy
           :param sentence_splitter: Callable that segments a document into sentences,
                                     defaults to scispacy
           """


        if type(base_path) == str:
            base_path: Path = Path(base_path)

        # column format
        columns = {0: "text", 1: "trigger", 2: "space-after"}

        # this dataset name
        dataset_name = self.__class__.__name__.lower()

        # default dataset folder is the cache root
        if not base_path:
            base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"

        if not (train_file.exists() and dev_file.exists()):
            train_folder, dev_folder, test_folder = biomedical.BIONLP2013_CG.download_corpus(data_folder / "original")
            train_data = self.parse_input_files(train_folder)
            dev_data = self.parse_input_files(dev_folder)

            sentence_splitter = SciSpacySentenceSplitter()

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)

        super(BIONLP2013_CG_TRIGGERS, self).__init__(
            data_folder, columns, tag_to_bioes="trigger", in_memory=in_memory
        )


    @staticmethod
    def parse_input_files(input_folder: Path) -> InternalBioNerDataset:
        documents = {}
        entities_per_document = {}

        for txt_file in input_folder.glob("*.txt"):
            name = txt_file.with_suffix("").name
            a1_file = txt_file.with_suffix(".a2")

            with txt_file.open() as f:
                documents[name] = f.read()

            with a1_file.open() as ann_reader:
                entities = []

                for line in ann_reader:
                    fields = line.strip().split("\t")
                    if fields[0].startswith("T"):
                        ann_type, start, end = fields[1].split()
                        entities.append(
                            Entity(
                                char_span=(int(start), int(end)), entity_type=ann_type
                            )
                        )
                entities_per_document[name] = entities

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()


    corpus = BIONLP2013_PC_TRIGGERS()
    tag_dictionary = corpus.make_tag_dictionary(tag_type="trigger")
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings("pubmed_pmc_wiki_sg_1M.gensim"),
        FlairEmbeddings('pm_pmc-forward/best-lm.pt'),
        FlairEmbeddings('pm_pmc-backward/best-lm.pt'),

    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='trigger',
                                            use_crf=True,
                                            locked_dropout=0.5)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    base_path = os.path.join("events/trigger_detection_models", corpus.__class__.__name__.lower())

    trainer.train(base_path=base_path, train_with_dev=False, max_epochs=100,
                  learning_rate=0.1, mini_batch_size=1)
