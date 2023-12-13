import torch
from torch.utils.data import Dataset

__all__ = [
    'DataManager',
    'TensorDataset'
    ]

class DataManager(object):
    def __init__(self, X, Y):
        self.X     = X
        self.Y     = Y
        self.index = 0

    def __len__(self):
        return len(self.Y)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.Y):
            raise StopIteration
        else:
            x = self.X[self.index]
            y = self.Y[self.index]
            self.index += 1
            return x, y

class TensorDataset(Dataset):
    def __init__(self, dataset):
        self.X = torch.FloatTensor(dataset.X)
        self.Y = torch.LongTensor(dataset.Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data   = self.X[index]
        target = self.Y[index]
        return data, target