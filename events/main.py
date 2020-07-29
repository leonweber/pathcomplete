import argparse
import json
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import WandbLogger

from events.model import EventExtractor



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')

    args = parser.parse_args()
    # args.config = Path("configs/gnn_bert.json")

    with args.config.open() as f:
        config = json.load(f)
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
                                          monitor="train_loss",
                                          mode="min",
                                          save_top_k=20)

    # model = EventExtractor.load_from_checkpoint("test2/epoch=23_v0.ckpt",
    #                                             config=config)
    model = EventExtractor(config=config)
    logger = []
    if not args.disable_wandb:
        logger.append(WandbLogger(project="events"))
    trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, check_val_every_n_epoch=1,
                         checkpoint_callback=checkpoint_callback, logger=logger, use_amp=True,
                         )
    trainer.fit(model)
