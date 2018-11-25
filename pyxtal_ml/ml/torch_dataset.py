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
        self.X_train = torch.from_numpy(self.xy[:, 0:-1]).float()
        self.Y_train = torch.from_numpy(self.xy[:, [-1]]).float()

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.len
    
class dataset_3d(data.Dataset):
    """
    Initiating a dataset for PyTorch using
    DataLoader method.

    Args:
        x: training feature.
        y: training property.
    """
    def __init__(self, x, y):
        self.len = y.shape[0]
        self.X_train = torch.from_numpy(x).float()
        self.Y_train = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

    def __len__(self):
        return self.len
