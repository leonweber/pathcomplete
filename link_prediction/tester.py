import torch
from eval_dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join
from sklearn.metrics import precision_recall_curve, average_precision_score


class Tester:
    def __init__(self, dataset, model_path, batch_size, split):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = dataset
        self.batch_size = batch_size
        self.split = split

    def test(self):
        all_ys = []
        all_scores = []
        last_batch = False
        while not last_batch:
            X, y = self.dataset.next_batch(self.batch_size, split=self.split)
            with torch.no_grad():
                all_scores.append(self.model.predict_relations(X).cpu().numpy()[:, 1:])
                all_ys.append(y.cpu().numpy())
            last_batch = self.dataset.was_last_batch(self.split)
        all_scores = np.concatenate(all_scores, axis=0)
        all_ys = np.concatenate(all_ys, axis=0)
        ap = average_precision_score(all_ys.ravel(), all_scores.ravel(), average='micro')

        return ap






    
    
