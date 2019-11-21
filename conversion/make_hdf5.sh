#!/usr/bin/env bash

#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/train.json distant_supervision/data/BioNLP-ST_2011/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/dev.json distant_supervision/data/BioNLP-ST_2011/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/test.json distant_supervision/data/BioNLP-ST_2011/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2011/all.json distant_supervision/data/BioNLP-ST_2011/all.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/train.json distant_supervision/data/BioNLP-ST_2013/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/dev.json distant_supervision/data/BioNLP-ST_2013/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/test.json distant_supervision/data/BioNLP-ST_2013/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/BioNLP-ST_2013/all.json distant_supervision/data/BioNLP-ST_2013/all.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
