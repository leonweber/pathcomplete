from sklearn.metrics import precision_recall_curve

from dataset import Dataset
from SimplE import SimplE
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class Trainer:
    def __init__(self, dataset, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = SimplE(dataset.num_ent, dataset.num_rel, args.emb_dim, self.device)
        self.dataset = dataset
        self.args = args
        
    def train(self):
        self.model.train()

        # optimizer = torch.optim.Adagrad(
        #     self.model.parameters(),
        #     lr=self.args.lr,
        #     weight_decay= 0,
        #     initial_accumulator_value= 0.1 #this is added because of the consistency to the original tensorflow code
        # )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        loss_fun = nn.BCEWithLogitsLoss()

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0

            all_scores = []
            all_ys = []

            while not last_batch:
                X, y = self.dataset.next_batch(self.args.batch_size, "train")
                last_batch = self.dataset.was_last_batch("train")
                optimizer.zero_grad()
                scores = self.model(X)
                loss = loss_fun(scores, y.float()) + (self.args.reg_lambda * self.model.l2_loss())
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

                all_scores.append(scores.cpu().detach())
                all_ys.append(y.cpu().detach())

            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.dataset.name + ")")
            precisions, recalls, _ = precision_recall_curve(np.concatenate(all_ys).ravel(),
                                                            np.concatenate(all_scores).ravel())
            ap = np.sum(np.diff(np.insert(recalls[::-1], 0, 0)) * precisions[::-1])
            print("ap: ", ap)
        
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + str(chkpnt) + ".chkpnt")

