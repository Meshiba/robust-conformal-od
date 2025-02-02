from torch.utils.data import Dataset
import torch


class cDataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.labels[idx]
