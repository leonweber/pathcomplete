import os

import argparse
from flair.datasets import biomedical
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, \
    FlairEmbeddings, PooledFlairEmbeddings, CharacterEmbeddings, FastTextEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v", required=True)
    parser.add_argument("--flair", required=True)

    args = parser.parse_args()


    corpus = biomedical.BIONLP2013_PC(entities_or_triggers="triggers")
    tag_dictionary = corpus.make_tag_dictionary(tag_type="ner")

    embedding_types = [

        WordEmbeddings(args.w2v),

        # uncomment in this line to use character embeddings
        # CharacterEmbeddings(),

        # comment in these lines to use flair embeddings
        PooledFlairEmbeddings(args.flair + '-forward/best-lm.pt', pooling='min'),
        PooledFlairEmbeddings(args.flair + '-backward/best-lm.pt', pooling='min'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    tagger: SequenceTagger = SequenceTagger(hidden_size=200,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type='ner',
                                            use_crf=True,
                                            locked_dropout=0.5)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    base_path = os.path.join("events/trigger_detection_models", corpus.__class__.__name__.lower())

    trainer.train(base_path=base_path, train_with_dev=False, max_epochs=100,
                  learning_rate=0.1, mini_batch_size=1)
