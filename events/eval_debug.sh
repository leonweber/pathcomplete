#!/usr/bin/env bash

for f in events/debug/eval/*a2; do
  python2 events/3rd_party/evaluation-PC.py $f -r events/data/BioNLP-ST_2013_PC_training_data/;
  read
done