import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import transformers

from distant_supervision.model import DistantBert


class DistantBertDataset(Dataset):

    def __init__(self, path):
        self.file = h5py.File(path)
        self.pairs = []
        for e1_id, e2_id in self.file['entity_ids']:
            e1 = self.file['id2entity'][e1_id].decode()
            e2 = self.file['id2entity'][e2_id].decode()
            self.pairs.append(f"{e1},{e2}")
        self.pairs = np.array(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        token_ids = self.file.get(f"token_ids/{pair}", np.array([-1]))[:]
        attention_masks = self.file.get(f"attention_masks/{pair}", np.array([-1]))[:]
        entity_pos = self.file.get(f"entity_positions/{pair}", np.array([-1]))[:]
        labels = self.file["labels"][idx]
        entity_ids = self.file["entity_ids"][idx]

        sample = {
            "token_ids": torch.from_numpy(token_ids).long(),
            "attention_masks": torch.from_numpy(attention_masks).long(),
            "entity_pos": torch.from_numpy(entity_pos).long(),
            "entity_ids": torch.from_numpy(entity_ids).long(),
            "labels": torch.from_numpy(labels).long(),
            "has_mentions": torch.tensor(token_ids[0] >= 0).bool()
        }

        return sample


train = DistantBertDataset(
    '/home/leon/projects/pathcomplete/distant_supervision/data/PathwayCommons11.reactome.hgnc.txt/train.hdf5.small')
dev = DistantBertDataset(
    '/home/leon/projects/pathcomplete/distant_supervision/data/PathwayCommons11.reactome.hgnc.txt/dev.hdf5.small')


train_loader = DataLoader(train, batch_size=1, shuffle=True)
distant_bert = DistantBert('/home/leon/data/scibert_scivocab_uncased')
distant_bert.cuda()

for i_batch, sample_batch in enumerate(train_loader):
    if sample_batch['has_mentions'].sum() > 0:
        _, x = distant_bert(sample_batch)


