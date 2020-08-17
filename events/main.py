import argparse
import json
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pprint import pprint

import torch
from flair.models import SequenceTagger
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
    parser.add_argument('--dev', action='store_true')
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

    if config["loss_weight_eg"] > 0:
        checkpoint_callback = ModelCheckpoint(filepath=args.output_dir,
                                              save_weights_only=True,
                                              verbose=True,
                                              monitor="val_f1",
                                              mode="max",
                                              save_top_k=1)
    else:
        checkpoint_callback = ModelCheckpoint(filepath=args.output_dir,
                                              save_weights_only=True,
                                              verbose=True,
                                              monitor="val_f1_td",
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

    if args.dev:
        latest_checkpoint = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
        latest_checkpoint = torch.load(latest_checkpoint)

        with Tee(args.output_dir/"test.log", "w"):
            model = EventExtractor(config=config)
            model.load_state_dict(latest_checkpoint["state_dict"], strict=False)
            # model.trigger_detector = SequenceTagger.load(config["trigger_detector"])
            trainer.test(model, model.val_dataloader())
            # model.validation_step = model.validation_step_gold
            # model.dev_dataset.predict = False
            # model.validation_epoch_end = model.validation_epoch_end_gold
            # trainer.test(model, model.val_gold_dataloader())

    if args.test:
        latest_checkpoint = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
        latest_checkpoint = torch.load(latest_checkpoint)

        with Tee(args.output_dir/"test.log", "w"):
            model = EventExtractor(config=config)
            model.load_state_dict(latest_checkpoint["state_dict"], strict=False)
            # model = EventExtractor.load_from_checkpoint(latest_checkpoint, config=config)
            # model.trigger_detector = SequenceTagger.load(config["trigger_detector"])
            trainer.test(model, model.test_dataloader())

    # best_checkpoint = list(args.output_dir.glob("*ckpt"))[0]
    # model = EventExtractor.load_from_checkpoint(best_checkpoint, config=config)
    # fname, text, ann = model.dev_dataset.predict_example_by_fname['PMID-12771181.txt']
    # a2_lines = model.predict(text, ann)
    # pass
