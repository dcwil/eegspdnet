import torch
import numpy as np

from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, features, labels, set_name):
        super(EEGDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.set_name = set_name

    def __getitem__(self, idx):
        data = torch.from_numpy(self.features[idx].copy()).float()
        label = np.asarray(self.labels[idx]).astype(np.compat.long)

        sample = {"data": data, "label": torch.from_numpy(label), "index": idx}
        return sample

    def __len__(self):
        return self.features.shape[0]
