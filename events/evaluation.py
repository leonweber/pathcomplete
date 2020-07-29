import os
import shlex
import shutil
from collections import defaultdict
from pathlib import Path
import random
from typing import Dict, List
import tempfile
import subprocess
import re
import torch
from ignite.metrics import Metric

from events import consts

THIRD_PARTY_DIR = Path(__file__).parent / '3rd_party'


class Evaluator:

    def __init__(self, eval_script: Path, data_dir: Path, result_re: str,
                 out_dir: Path, verbose: bool = True):
        self.eval_script = eval_script
        self.data_dir = data_dir
        self.result_re = result_re
        self.verbose = verbose
        self.out_dir = out_dir

    def evaluate(self, predicted_a2: Dict[str, List[str]]):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)
        # with tempfile.TemporaryDirectory() as d:
        for fname, preds in predicted_a2.items():
            with (Path(self.out_dir) / fname).with_suffix('.a2').open('w') as f:
                f.write(preds)

        cmd = f'python2 {THIRD_PARTY_DIR / self.eval_script} -r {self.data_dir} {" ".join([str(i) for i in Path(self.out_dir).glob("*a2")])}'
        result = subprocess.run(cmd, capture_output=True, shell=True)
        for line in result.stdout.splitlines():
            line = line.decode()
            if self.verbose:
                print(line)
            match = re.match(self.result_re, line)
            if match:
                p, r, f = match.group(1), match.group(2), match.group(3)
        try:
            return {'precision': float(p), 'recall': float(r), 'f1': float(f)}
        except UnboundLocalError:
            print("Error in evaluation:")
            print(result.stderr)
            return {'precision': 0., 'recall': 0., 'f1': 0.}


class BioNLPMetric(Metric):
    def __init__(self, evaluator, output_transform=lambda x: x,
                 key='f1'):
        self.evaluator = evaluator
        self.key = key
        self._required_output_keys = None

        super(BioNLPMetric, self).__init__(output_transform=output_transform, device='cpu')

    def reset(self):
        self._predictions = {}

    def update(self, output):
        pred, batch = output
        self._predictions[batch['fname']] = pred

    def compute(self):
        return self.evaluator.evaluate(self._predictions)[self.key]


def output_transform_edges(output):
    all_preds = []
    all_trues = []
    for logits in output[0]['aux']:
        n_classes = logits.size(-1)
        all_preds.append(logits.view(-1, n_classes))
    all_preds = torch.cat(all_preds)

    for labels in output[1]['labels']:
        all_trues += labels
    all_trues = torch.tensor(all_trues)

    return all_preds, all_trues

class Acc(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.n_correct = 0
        self.n_pred = 0
        super(Acc, self).__init__(output_transform=output_transform, device='cpu')

    def reset(self):
        self.n_correct = 0
        self.n_pred = 0

    def update(self, output):
        pred, true = output
        self.n_correct += (pred == true).sum().item()
        self.n_pred += len(pred)

    def compute(self):
        return self.n_correct / self.n_pred

class F1(Metric):
    def __init__(self, output_transform=lambda x: x):
        self.n_tp = 0
        self.n_fp = 0
        self.n_fn = 0
        super(F1, self).__init__(output_transform=output_transform, device='cpu')

    def reset(self):
        self.n_tp = 0
        self.n_fp = 0
        self.n_fn = 0

    def update(self, output):
        pass


    def compute(self):
        return 2*self.n_tp / (2*self.n_tp + self.n_fn + self.n_fp)
