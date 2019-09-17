import torch
from dataset import Dataset
import numpy as np
from measure import Measure
from os import listdir
from os.path import isfile, join
from sklearn.metrics import precision_recall_curve

class Tester:
    def __init__(self, dataset, model_path, batch_size, split):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location = self.device)
        self.model.eval()
        self.dataset = Dataset(dataset)
        self.batch_size = batch_size
        self.split = split

    def test(self):
        last_batch = False
        aps = []
        while not last_batch:
            X, y = self.dataset.next_batch(self.batch_size, split=self.split)
            with torch.no_grad():
                scores = self.model(X).cpu().numpy()
                precisions, recalls, _ = precision_recall_curve(y.ravel(), scores.ravel())
                ap = np.sum(np.diff(np.insert(recalls[::-1], 0, 0)) * precisions[::-1])
                aps.append(ap)

        return np.mean(aps)






    
    