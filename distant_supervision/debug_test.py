import json
import shutil
import sys
import os

from allennlp.commands import main


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "predict",
    "runs/run05/model.tar.gz",
    "data/PathwayCommons11.reactome.hgnc.txt/dev.json.small",
    "--include-package", "relex",
    "--cuda-device", "0",
    "--batch-size", "32",
    "--use-dataset-reader",
    "--predictor", "relex",
    "--output-file", "debug-preds.json",
    "--silent"

]

main()