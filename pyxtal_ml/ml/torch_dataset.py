import torch
from torch.utils import data

class dataset(data.Dataset):
    """
    Initiating a dataset for PyTorch using
    DataLoader method.

    Args:
        xy: training dataset.
    """
    def __init__(self, xy):
        self.xy = xy
        self.len = xy.shape[0]
        self.X_train = torch.from_numpy(self.xy[:, 0:-1])
        self.Y_train = torch.from_numpy(self.xy[:, [-1]])

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.len
