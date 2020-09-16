#!/usr/bin/env bash

#for f in events/debug_small/eval/*a2; do
for f in events/runs/self_loops/eval/*a2; do
#  python2 events/3rd_party/evaluation-PC.py $f -r events/data/BioNLP-ST_2013_PC_training_data/ -v;
  python2 events/3rd_party/evaluation-PC.py $f -r events/data/BioNLP-ST_2013_PC_development_data/ -v;
  read
done