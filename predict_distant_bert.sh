#!/usr/bin/env bash
runs=( bionlp13_01 bionlp13_02 bionlp13_03 bionlp13_04 bionlp13_05 bionlp13_06 bionlp13_07 bionlp13_08 bionlp13_09 bionlp13_10)
#data=PathwayCommons11.pid.hgnc.txt
#data=p53
data=BioNLP-ST_2013
type=test

for run in ${runs[@]}; do
python -m distant_supervision.predict_bert distant_supervision/data/$data/"$type"_masked.hdf5 distant_supervision/runs/$run/"$data"_"$type"_preds.txt --model_path distant_supervision/runs/$run --data distant_supervision/data/$data/"$type".json --device cuda;
done