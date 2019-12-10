#!/usr/bin/env bash

python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json
python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json

python -m conversion.ds_tag_entities distant_supervision/data/NFKB/test.json distant_supervision/data/NFKB/test_masked.json

#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

#python -m conversion.ds_to_hdf5 distant_supervision/data/NFKB/test.json distant_supervision/data/NFKB/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/NFKB/test_masked.json distant_supervision/data/NFKB/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

