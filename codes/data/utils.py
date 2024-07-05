
import numpy as np

from data.types import DataManager
import data.config as cfg

def load_raw_dataset(path="../../dataset_kichang", dataset='mitbih_arr') -> tuple[DataManager, DataManager]:
    X = np.load(f'{path}/{dataset}/x_train.npy') # (75455, 1, 1800) float64
    Y = np.load(f'{path}/{dataset}/y_train.npy') # (75455, ) int32
    trainset = DataManager(X, Y)
    X = np.load(f'{path}/{dataset}/x_test.npy') # (18864, 1, 1800)
    Y = np.load(f'{path}/{dataset}/y_test.npy') # (18864,)
    testset = DataManager(X, Y)
    return trainset, testset

def select_n_shots_per_class(trainset, dataset, n_shots) -> DataManager:
    """
    From the trainset, choose n_shots samples for each class.
    """
    
    X = []
    Y = np.array([])
    for i in range(cfg.N_CLASS[dataset]):
        idx = np.random.choice(len(trainset.Y[trainset.Y==i]), n_shots, replace=False)
        X.append(trainset.X[trainset.Y==i][idx])
        Y = np.hstack([Y, np.ones(n_shots)*i])
    X = np.concatenate(X, axis=0)
    trainset = DataManager(X, Y)
    return trainset