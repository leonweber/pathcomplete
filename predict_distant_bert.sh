#!/usr/bin/env bash
run=run201
#data=PathwayCommons11.pid.hgnc.txt
data=p53
type=test
python -m distant_supervision.predict_bert distant_supervision/data/$data/"$type"_masked.hdf5 distant_supervision/runs/$run/"$data"_"$type"_preds.txt --model_path distant_supervision/runs/$run --data distant_supervision/data/$data/$type.json --device cuda