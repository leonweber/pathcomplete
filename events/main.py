import argparse
import json
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pprint import pprint

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import WandbLogger

from events.model import EventExtractor
from util.utils import Tee

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    with args.config.open() as f:
        config = json.load(f)
    config["output_dir"] = args.output_dir
    pprint(config)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(args.config, args.output_dir)

    checkpoint_callback = ModelCheckpoint(filepath=args.output_dir,
                                          save_weights_only=True,
                                          verbose=True,
                                          monitor="val_f1",
                                          mode="max",
                                          save_top_k=1)


    logger = []
    if not args.disable_wandb:
        logger.append(WandbLogger(project="events"))
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, check_val_every_n_epoch=1,
                         checkpoint_callback=checkpoint_callback, logger=logger, use_amp=True,
                         )
    if args.train:
        with Tee(args.output_dir/"train.log", "w"):
            model = EventExtractor(config=config)
            trainer.fit(model)

    if args.test:
        best_checkpoint = list(args.output_dir.glob("*ckpt"))[0]
        with Tee(args.output_dir/"test.log", "w"):
            model = EventExtractor.load_from_checkpoint(best_checkpoint, config=config)
            trainer.test(model, model.val_dataloader())
    # best_checkpoint = list(args.output_dir.glob("*ckpt"))[0]
    # model = EventExtractor.load_from_checkpoint(best_checkpoint, config=config)
    # fname, text, ann = model.dev_dataset.predict_example_by_fname['PMID-12771181.txt']
    # a2_lines = model.predict(text, ann)
    # pass
