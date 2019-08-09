#!/usr/bin/env bash

wget https://www.pathwaycommons.org/archives/PC2/v11/PathwayCommons11."$1".hgnc.txt.gz
gunzip PathwayCommons11."$1".hgnc.txt.gz
mv PathwayCommons11."$1".hgnc.txt ../data
python make_dataset.py ../data/PathwayCommons11."$1".hgnc.txt --small
python json_to_link_prediction.py --name "$1"
