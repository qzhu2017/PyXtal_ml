import torch
from torch.utils import data

class dataset(data.Dataset):
    """
    Building a dataset for PyTorch using
    DataLoader method.

    Args:
        xy: training dataset.
    """
    def __init__(self, xy):
        self.xy = xy
        self.len = xy.shape[0]
        self.x_train = torch.from_numpy(self.xy[:, 0:-1])
        self.y_train = torch.from_numpy(self.xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len
