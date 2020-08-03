#!/usr/bin/env bash

#runs=( bionlp13_01 bionlp13_02 bionlp13_03 bionlp13_04 bionlp13_05 bionlp13_06 bionlp13_07 bionlp13_08 bionlp13_09 bionlp13_10 )
runs=( run029 )
#data=PathwayCommons13.pid.hgnc.txt
#data=p53
data=BioNLP-ST_2011
type=test

for run in ${runs[@]}; do

allennlp predict --include-package relex --output-file old_runs/"$run"/"$data"_"$type"_preds.txt old_runs/$run/model.tar.gz data/$data/"$type"_masked.json --predictor relex --use-dataset-reader -o '{"dataset_reader": {"with_metadata": true}}' --silent --weights-file old_runs/$run/best.th

done