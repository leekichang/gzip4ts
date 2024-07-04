import numpy as np

from ..data.types import DataManager
from ..data import config as cfg

def load_dataset(path="../dataset", dataset='mitbih_arr', sample=(-1, -1), generate=False):
    """
    TODO: 없으면 만들고 있으면 불러오기
    """
    
    
    if generate:
        # X = np.load(f'{path}/{dataset}/x_train.npy')
        # Y = np.load(f'{path}/{dataset}/y_train.npy') # (18864,)
        # print(len(X))
        # random_idx = np.random.choice(len(X), 2000, replace=False)
        # X = X[random_idx]
        # Y = Y[random_idx]
        # np.savez_compressed("./dataset/inference_dataset.npz", X=X, Y=Y)
        data = np.load("./dataset/inference_dataset.npz")
        X = data["X"]
        Y = data["Y"]
    else:
        X = np.load(f'{path}/{dataset}/x_train.npy') # (75455, 1, 1800) float64
        Y = np.load(f'{path}/{dataset}/y_train.npy') # (75455, ) int32
    trainset = DataManager(X, Y)
    if generate:
        data = np.load("./dataset/inference_dataset.npz")
        X = data["X"]
        Y = data["Y"]
    else:
        X = np.load(f'{path}/{dataset}/x_test.npy') # (18864, 1, 1800)
        Y = np.load(f'{path}/{dataset}/y_test.npy') # (18864,)
    testset = DataManager(X, Y)
    return trainset, testset

def choose_trainset(trainset, args):
    np.random.seed(args.seed)
    X = []
    Y = np.array([])
    for i in range(cfg.N_CLASS[args.dataset]):
        idx = np.random.choice(len(trainset.Y[trainset.Y==i]), args.n_shots, replace=False)
        X.append(trainset.X[trainset.Y==i][idx])
        Y = np.hstack([Y, np.ones(args.n_shots)*i])
    X = np.concatenate(X, axis=0)
    trainset = DataManager(X, Y)
    return trainset