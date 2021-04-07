import argparse
from collections import defaultdict
import json
import logging
import os
import shutil
from pathlib import Path
import numpy as np

import pytorch_lightning as pl
from pprint import pprint

import torch
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from events import consts, dataset
from events.evaluation import Evaluator
from events.model import EventExtractor
from util.utils import Tee

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--eval_dev', action='store_true')
    parser.add_argument('--eval_dev_steps', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--model', nargs="*")

    args = parser.parse_args()

    with args.config.open() as f:
        config = json.load(f)
    config["output_dir"] = args.output_dir
    pprint(config)

    config["small"] = args.small

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(args.config, args.output_dir)

    if not config["small"]:
        checkpoint_callback = ModelCheckpoint(filepath=args.output_dir,
                                                  save_weights_only=True,
                                                  verbose=True,
                                                  monitor="val_acc",
                                                  mode="max",
                                                  save_top_k=1)
    else:
        checkpoint_callback = None


    logger = []
    pl.seed_everything(args.seed)
    if not args.disable_wandb:
        logger.append(WandbLogger(project="events"))
    if args.train:
        trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, max_epochs=config["num_epochs"],
                            checkpoint_callback=checkpoint_callback, logger=logger, precision=16,
                            track_grad_norm=-1, num_sanity_val_steps=0, check_val_every_n_epoch=1
                            )
        # with Tee(args.output_dir/"train.log", "w"):
        if args.resume:
            if not args.model:
                 args.model = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
            model = EventExtractor.load_from_checkpoint(str(args.model), config=config)
        else:
            model = EventExtractor(config=config)
        trainer.fit(model)

    if args.dev:
        with Tee(args.output_dir/"test.log", "w"):
            if not args.model:
                args.model = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
            model = EventExtractor.load_from_checkpoint(str(args.model), config=config)
            trainer = pl.Trainer(gpus=1, accumulate_grad_batches=1, max_epochs=config["num_epochs"],
                                checkpoint_callback=checkpoint_callback, logger=logger, precision=16,
                                track_grad_norm=-1, num_sanity_val_steps=0, check_val_every_n_epoch=1
                                )
            trainer.test(model, model.val_dataloader())


    if args.eval_dev_steps:
        with Tee(args.output_dir/"test.log", "w"):
            correct = []
            if not args.model:
                args.model = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0:1]

            models = [EventExtractor.load_from_checkpoint(str(m), config=config) for m in args.model]
            for model in models:
                model.eval()
                model.cuda()
            tps = defaultdict(int)
            fps = defaultdict(int)
            fns = defaultdict(int)
            correct = []
            for text, G_true in tqdm(list(models[0].dev_dataset.text_to_graph.items())):
                # known_nodes = []
                known_nodes = model.dev_dataset.node_order(G_true)[:-1]
                for node, data in G_true.nodes(data=True):
                    if data["type"] == "entity":
                        known_nodes.append(node)

                        for u, _, edge_data in G_true.in_edges(node, data=True):
                            if edge_data["type"] == "self":
                                known_nodes.append(u)
                G_known = G_true.subgraph(known_nodes)
                model = models[0]

                for node in model.train_dataset.node_order(G_true):
                    if node not in known_nodes:
                        break

                # stmts_preds = []
                G_true = G_true.subgraph(known_nodes + [node])
                G_pred = model.predict(text, G_known)
                all_stmts_true = [i for i in model.dev_dataset.graph_to_statements(G_true) if i not in known_nodes]
                try:
                    all_stmts_pred = [i for i in model.dev_dataset.graph_to_statements(G_pred) if i not in known_nodes]
                except:
                    all_stmts_pred = []
                if all_stmts_pred:
                    correct.append(all_stmts_pred[0] == all_stmts_true[0])
                else:
                    correct.append(False)
                print(np.mean(correct))
                for i in range(0, max(len(all_stmts_true), len(all_stmts_pred))):
                    stmts_true = all_stmts_true[i:i+1]
                    stmts_pred = all_stmts_pred[i:i+1]

                    # try:
                    # for model in models:
                    #     try:
                    #         G_pred = model.predict(text)
                    #         stmts_preds.append(set(model.dev_dataset.graph_to_statements(G_pred)))
                    #     except IndexError:
                    #         continue
                    
                    # stmts_pred = stmts_preds[0]
                    # for pred in stmts_preds:
                    #     stmts_pred = stmts_pred & pred
                    # majority_stmts = []
                    # for stmt in stmts_pred:
                    #     n_hits = sum(int(stmt in stmts) for stmts in stmts_preds)
                    #     if n_hits > len(models)//2:
                    #         majority_stmts.append(stmt)
                    # except IndexError:
                        # stmts_pred = set()

                    tps[i] += len(set(all_stmts_true) & set(stmts_pred))
                    fps[i] += len(set(stmts_pred) - set(all_stmts_true))
                    fns[i] += len(set(stmts_true) - set(all_stmts_pred))

                for i in sorted(tps):
                    try:
                        p = tps[i] / (tps[i] + fps[i])
                    except ZeroDivisionError:
                        p = 0
                    try:
                        r = tps[i] / (tps[i] + fns[i])
                    except ZeroDivisionError:
                        r = 0
                    try:
                        f1 = 2*p*r / (p + r)
                    except ZeroDivisionError:
                        f1 = 0
                    print(f"{i}: P {p*100}, R {r*100}, F1 {f1}")
                print()

    if args.test:
        with Tee(args.output_dir/"test.log", "w"):
            correct = []
            if not args.model:
                args.model = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
            model = EventExtractor.load_from_checkpoint(str(args.model), config=config)
            model.eval()
            model.cuda()
            tps = 0
            fps = 0
            fns = 0
            DSType = getattr(dataset, config["dataset_type"])
            ds = DSType(config["test"], config["bert"])
            for text, G_true in tqdm(list(ds.text_to_graph.items())):
                stmts_true = set(ds.graph_to_statements(G_true))
                # try:
                G_pred = model.predict(text)
                stmts_pred = set(ds.graph_to_statements(G_pred))
                # except IndexError:
                    # stmts_pred = set()

                tps += len(stmts_true & stmts_pred)
                fps += len(stmts_pred - stmts_true)
                fns += len(stmts_true - stmts_pred)

                try:
                    p = tps / (tps + fps)
                except ZeroDivisionError:
                    p = 0
                try:
                    r = tps / (tps + fns)
                except ZeroDivisionError:
                    r = 0
                try:
                    f1 = 2*p*r / (p + r)
                except ZeroDivisionError:
                    f1 = 0
                print(f"P {p*100}, R {r*100}, F1 {f1}")



            # trainer.test(model, model.val_dataloader())

    if args.eval_train:
        with Tee(args.output_dir/"test.log", "w"):
            correct = []
            if not args.model:
                args.model = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
            model = EventExtractor.load_from_checkpoint(str(args.model), config=config)
            model.eval()
            tps = 0
            fps = 0
            fns = 0
            for text, G_true in tqdm(list(model.train_dataset.text_to_graph.items())):
                stmts_true = set(model.dev_dataset.graph_to_statements(G_true))
                try:
                    G_pred = model.predict(text)
                    stmts_pred = set(model.train_dataset.graph_to_statements(G_pred))
                except IndexError:
                    stmts_pred = set()

                tps += len(stmts_true & stmts_pred)
                fps += len(stmts_pred - stmts_true)
                fns += len(stmts_true - stmts_pred)

                p = tps / (tps + fps)
                r = tps / (tps + fns)
                try:
                    f1 = 2*p*r / (p + r)
                except ZeroDivisionError:
                    f1 = 0
                print(f"P {p*100}, R {r*100}, F1 {f1}")

    
    if args.predict:
        if not args.model:
            args.model = sorted(args.output_dir.glob("*ckpt"), key=os.path.getctime)[::-1][0]
        config["small"] = True
        model = EventExtractor.load_from_checkpoint(str(args.model), config=config)
        model.eval()
        model.cuda()
        # model.predict("Heparin cofactor II (HCII) is a highly specific serine proteinase inhibitor, which complexes covalently with thrombin in a reaction catalyzed by heparin and other polyanions.")
        # model.predict("Progressive accumulation of a cytotoxic metabolite, galactosylsphingosine (psychosine), was found in the brain of the twitcher mouse, a mutant caused by genetic deficiency of galactosylceramidase.")
        model.predict("Acidic fibroblast growth factor (FGF-1), a prototype member of the heparin-binding growth factor family, influences proliferation, differentiation, and protein synthesis in different cell types")
        with open("events/data/BioCreative_BEL_Track/SampleSet.tab") as f:
            texts = [l.split("\t")[2] for l in f][1:]
        with open("foo.txt", "w") as f:
            f.write("")
        for i, text in tqdm(list(enumerate(texts))):
            with open("foo.txt", "a") as f:
                # try:
                with torch.no_grad():
                    G = model.predict(text)
                preds = []
                for node in G.nodes:
                    if len(G.in_edges(node)) == 0:
                        preds.append(model.train_dataset.node_to_bel(node, G))
                # except IndexError:
                #     preds = []
                f.write(f"{i}: {' '.join(preds)}\n")


    # best_checkpoint = list(args.output_dir.glob("*ckpt"))[0]
    # model = EventExtractor.load_from_checkpoint(best_checkpoint, config=config)
    # fname, text, ann = model.dev_dataset.predict_example_by_fname['PMID-12771181.txt']
    # a2_lines = model.predict(text, ann)
    # pass
