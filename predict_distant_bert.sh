#!/usr/bin/env bash
runs=( run037 run038 run041 run043 )
#data=PathwayCommons11.pid.hgnc.txt
#data=p53
data=BioNLP-ST_2011
type=test

for run in ${runs[@]}; do
python -m distant_supervision.predict_bert distant_supervision/data/$data/"$type"_masked.hdf5 distant_supervision/runs/$run/"$data"_"$type"_preds.txt --model_path distant_supervision/runs/$run --data distant_supervision/data/$data/"$type".json --device cuda;
done