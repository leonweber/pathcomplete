#!/usr/bin/env bash

#python conversion/filter_ds_annotations.py distant_supervision/data/nfkb/test.json distant_supervision/data/nfkb/test_nodb.json --triple_blacklist data/PathwayCommons11.pid.hgnc.txt.json data/PathwayCommons11.reactome.hgnc.txt.json
#python conversion/filter_ds_annotations.py distant_supervision/data/p53/test.json distant_supervision/data/p53/test_nodb.json --triple_blacklist data/PathwayCommons11.pid.hgnc.txt.json data/PathwayCommons11.reactome.hgnc.txt.json

#python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json
#python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json
#python -m conversion.ds_tag_entities distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json

#python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/train_masked.json --direct_data distant_supervision/data/BioNLP-STs/all_masked.json --pair_blacklist distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json
#python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json
#python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/test_masked.json

#python -m conversion.ds_tag_entities distant_supervision/data/nfkb/test.json distant_supervision/data/nfkb/test_masked.json
#python -m conversion.ds_tag_entities distant_supervision/data/p53/test.json distant_supervision/data/p53/test_masked.json

#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/train_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/dev_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#
#python -m conversion.ds_to_hdf5 distant_supervision/data/nfkb/test.json distant_supervision/data/nfkb/test.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
#python -m conversion.ds_to_hdf5 distant_supervision/data/p53/test_masked.json distant_supervision/data/p53/test_masked.hdf5 --tokenizer ~/data/scibert_scivocab_uncased

#python conversion/ds_filter_by_year.py distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json --max_year 2012
#python -m conversion.ds_to_hdf5 distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.hdf5 --tokenizer ~/data/scibert_scivocab_uncased
python -m conversion.ds_to_comb_dist_relex distant_supervision/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json distant_supervision/comb_dist_direct_relex/data/PathwayCommons11.pid.hgnc.txt/test_masked_2012.json
