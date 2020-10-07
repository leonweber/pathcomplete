#!/usr/bin/env bash

#for f in events/debug_small/eval/*a2; do
for f in events/runs/self_loops/eval/*a2; do
#for f in events/runs/devel/eval_train3/*a2; do
#for f in events/runs/ge13/eval/*a2; do
#for f in events/foo/*a2; do
    echo $f;
#  python2 events/3rd_party/evaluation-PC.py $f -r events/data/BioNLP-ST_2013_PC_training_data/ -v;
  python2 events/3rd_party/evaluation-PC.py $f -r events/data/BioNLP-ST_2013_PC_development_data/ -v;
#   perl events/3rd_party/a2-evaluate.pl -g events/data/BioNLP-ST-2013_GE_devel_data_rev3/ -s -p $f;
  read
done