import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DistantBertDataset(Dataset):

    def __init__(self, path, max_bag_size=None, max_length=512, ignore_no_mentions=False):
        self.file = h5py.File(path, 'r')
        self.max_bag_size = max_bag_size
        self.max_length = max_length
        self.pairs = []
        for e1_id, e2_id in self.file['entity_ids']:
            e1 = self.file['id2entity'][e1_id].decode()
            e2 = self.file['id2entity'][e2_id].decode()
            self.pairs.append(f"{e1},{e2}")
        self.pairs = np.array(self.pairs)
        if ignore_no_mentions:
            filtered_pairs = []
            for pair in self.pairs:
                if pair in self.file['token_ids']:
                    filtered_pairs.append(pair)
            self.pairs = filtered_pairs

        self.n_classes = len(self.file['id2label'])
        self.n_entities = len(self.file['id2entity'])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pair = self.pairs[idx]
        token_ids = self.file.get(f"token_ids/{pair}", np.array([[-1]]))[:]
        attention_masks = self.file.get(f"attention_masks/{pair}", np.array([[-1]]))[:]
        entity_pos = self.file.get(f"entity_positions/{pair}", np.array([[-1]]))[:] # bag_size x e1/e2 x start/end
        pmids = self.file.get(f"pmids/{pair}", np.array([[-1]]))[:] # bag_size x e1/e2 x start/end
        labels = self.file["labels"][idx]
        entity_ids = self.file["entity_ids"][idx]

        token_ids = token_ids[:self.max_bag_size, :self.max_length]
        attention_masks = attention_masks[:self.max_bag_size, :self.max_length]
        entity_pos = entity_pos[:self.max_bag_size]


        sample = {
            "token_ids": torch.from_numpy(token_ids).long(),
            "attention_masks": torch.from_numpy(attention_masks).long(),
            "entity_pos": torch.from_numpy(entity_pos).long(),
            "entity_ids": torch.from_numpy(entity_ids).long(),
            "labels": torch.from_numpy(labels).long(),
            "has_mentions": torch.tensor([token_ids[0][0] >= 0]).bool(),
            "pmids": torch.tensor(pmids).long()
        }

        return sample