import shlex
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
                 verbose: bool = False):
        self.eval_script = eval_script
        self.data_dir = data_dir
        self.result_re = result_re
        self.verbose = verbose

    def evaluate(self, predicted_a2: Dict[str, List[str]]):
        with tempfile.TemporaryDirectory() as d:
            for fname, preds in predicted_a2.items():
                with (Path(d) / fname).with_suffix('.a2').open('w') as f:
                    f.writelines(p + "\n" for p in preds)

            cmd = f'python2 {THIRD_PARTY_DIR / self.eval_script} -r {self.data_dir} {" ".join([str(i) for i in Path(d).glob("*a2")])}'
            result = subprocess.run(cmd, capture_output=True, shell=True)
            for line in result.stdout.splitlines():
                line = line.decode()
                if self.verbose:
                    print(line)
                match = re.match(self.result_re, line)
                if match:
                    p, r, f = match.group(1), match.group(2), match.group(3)
        return {'precision': float(p), 'recall': float(r), 'f1': float(f)}


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
        self._predictions[output[1]['fname']] = output[0]['a2']

    def compute(self):
        return self.evaluator.evaluate(self._predictions)[self.key]


def output_transform_event_node(output):
    all_preds = output[0]['aux']['node_logits']
    all_trues = output[1]['node_targets']

    # score only supports one index with 1
    # if prediction is true, set predicted index to 1, otherwise set any predicted index to 1
    pred_idcs = []
    true_idcs = []

    for pred, true in zip(all_preds, all_trues):
        pred_idx = pred.argmax()
        some_true_idx = true.argmax()
        is_true = true[pred_idx].bool().item()

        pred_idcs.append(pred_idx)
        if is_true:
            true_idcs.append(pred_idx)
        else:
            true_idcs.append(some_true_idx)

    return torch.tensor(pred_idcs), torch.tensor(true_idcs)

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
