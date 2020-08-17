import os

import argparse
from collections import defaultdict
from copy import deepcopy
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
from flair.file_utils import cached_path, Tqdm, unpack_file


def merge_entities_with_same_span(data):
    merged_entities = {}
    for doc, entities in data.entities_per_document.items():
        new_entities = []
        entities_by_span = defaultdict(list)
        for e in entities:
            entities_by_span[e.char_span].append(e)
        for span, span_entities in entities_by_span.items():
            span = (span.start, span.stop)
            new_type = "/".join(sorted([i.type for i in span_entities]))
            new_entities.append(Entity(span, new_type))
        merged_entities[doc] = new_entities

    return InternalBioNerDataset(documents=deepcopy(data.documents),
                                 entities_per_document=merged_entities)



class BioNLPCorpus(ColumnCorpus):
    def download_corpus(self):
        raise NotImplementedError

    def __init__(
            self,
            dataset_name
    ):

        # column format
        columns = {0: "text", 1: "trigger", 2: "space-after"}

        # default dataset folder is the cache root
        base_path = Path(flair.cache_root) / "datasets"
        data_folder = base_path / dataset_name

        train_file = data_folder / "train.conll"
        dev_file = data_folder / "dev.conll"

        if not (train_file.exists() and dev_file.exists()):
            train_folder, dev_folder, test_folder = self.download_corpus(data_folder / "original")
            train_data = self.parse_input_files(train_folder)
            dev_data = self.parse_input_files(dev_folder)

            # train_data = merge_entities_with_same_span(train_data)
            # dev_data = merge_entities_with_same_span(dev_data)

            sentence_splitter = SciSpacySentenceSplitter()

            conll_writer = CoNLLWriter(sentence_splitter=sentence_splitter)
            conll_writer.write_to_conll(train_data, train_file)
            conll_writer.write_to_conll(dev_data, dev_file)

        super(BioNLPCorpus, self).__init__(
            data_folder, columns, tag_to_bioes="trigger", in_memory=True
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
                                char_span=(int(start), int(end)), entity_type="Trigger"
                            )
                        )
                entities_per_document[name] = entities

        return InternalBioNerDataset(
            documents=documents, entities_per_document=entities_per_document
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


class BIONLP2013_PC_TRIGGERS(BioNLPCorpus):
    def __init__(self):
        dataset_name = self.__class__.__name__.lower()
        super().__init__(dataset_name)

    @staticmethod
    def download_corpus(download_folder: Path) -> Tuple[Path, Path, Path]:
        train_url = (
            "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz"
        )
        dev_url = (
            "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz"
        )
        test_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_test_data.tar.gz"

        cached_path(train_url, download_folder)
        cached_path(dev_url, download_folder)
        cached_path(test_url, download_folder)

        unpack_file(
            download_folder / "BioNLP-ST_2013_PC_training_data.tar.gz",
            download_folder,
            keep=False,
            )
        unpack_file(
            download_folder / "BioNLP-ST_2013_PC_development_data.tar.gz",
            download_folder,
            keep=False,
            )
        unpack_file(
            download_folder / "BioNLP-ST_2013_PC_test_data.tar.gz",
            download_folder,
            keep=False,
            )

        train_folder = download_folder / "BioNLP-ST_2013_PC_training_data"
        dev_folder = download_folder / "BioNLP-ST_2013_PC_development_data"
        test_folder = download_folder / "BioNLP-ST_2013_PC_test_data"

        return train_folder, dev_folder, test_folder


class BIONLP2013_CG_TRIGGERS(BioNLPCorpus):
    def __init__(self):
        dataset_name = self.__class__.__name__.lower()
        super().__init__(dataset_name)

    @staticmethod
    def download_corpus(download_folder: Path) -> Tuple[Path, Path, Path]:
        train_url = (
            "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_training_data.tar.gz"
        )
        dev_url = (
            "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_development_data.tar.gz"
        )
        test_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_test_data.tar.gz"

        download_folder = download_folder / "original"

        cached_path(train_url, download_folder)
        cached_path(dev_url, download_folder)
        cached_path(test_url, download_folder)

        unpack_file(
            download_folder / "BioNLP-ST_2013_CG_training_data.tar.gz",
            download_folder,
            keep=False,
            )
        unpack_file(
            download_folder / "BioNLP-ST_2013_CG_development_data.tar.gz",
            download_folder,
            keep=False,
            )
        unpack_file(
            download_folder / "BioNLP-ST_2013_CG_test_data.tar.gz",
            download_folder,
            keep=False,
            )

        train_folder = download_folder / "BioNLP-ST_2013_CG_training_data"
        dev_folder = download_folder / "BioNLP-ST_2013_CG_development_data"
        test_folder = download_folder / "BioNLP-ST_2013_CG_test_data"

        return train_folder, dev_folder, test_folder

class BIONLP2013_GE_TRIGGERS(BioNLPCorpus):
    def __init__(self):
        dataset_name = self.__class__.__name__.lower()
        super().__init__(dataset_name)

    @staticmethod
    def download_corpus(download_folder: Path) -> Tuple[Path, Path, Path]:
        train_url = (
            "http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0"
        )
        dev_url = (
            "http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0"
        )
        test_url = "http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_test_data_rev1.tar.gz?attredirects=0"

        download_folder = download_folder / "original"

        cached_path(train_url, download_folder)
        cached_path(dev_url, download_folder)
        cached_path(test_url, download_folder)

        unpack_file(
            download_folder /"BioNLP-ST-2013_GE_train_data_rev3.tar.gz?attredirects=0" ,
            download_folder,
            keep=False,
            mode="targz"
            )
        unpack_file(
            download_folder / "BioNLP-ST-2013_GE_devel_data_rev3.tar.gz?attredirects=0",
            download_folder,
            keep=False,
            mode="targz"
            )
        unpack_file(
            download_folder / "BioNLP-ST-2013_GE_test_data_rev1.tar.gz?attredirects=0",
            download_folder,
            keep=False,
            mode="targz"
            )

        train_folder = download_folder / "BioNLP-ST-2013_GE_train_data_rev3"
        dev_folder = download_folder / "BioNLP-ST-2013_GE_devel_data_rev3"
        test_folder = download_folder / "BioNLP-ST-2013_GE_test_data_rev3"

        return train_folder, dev_folder, test_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()


    corpus = BIONLP2013_PC_TRIGGERS()
    # corpus = BIONLP2013_GE_TRIGGERS()
    # print(sorted([len(s.to_original_text()) for s in corpus.get_all_sentences()])[::-1])
    # for s in corpus.get_all_sentences():
    #     if len(s.to_original_text()) > 500:
    #         print(s.to_original_text())
    # corpus.filter_long_sentences(500)
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
    base_path = os.path.join("events/trigger_detection_models/no_types", corpus.__class__.__name__.lower())

    trainer.train(base_path=base_path, train_with_dev=False, max_epochs=100,
                  learning_rate=0.1, mini_batch_size=32)


