import os
import pickle
import argparse
import numpy as np

import models
import codes.DataConfig as cfg
from DataManager import DataManager

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    # default arguments
    parser.add_argument('--exp_name'  , help='quant_str'         , type=str     , required=True)
    parser.add_argument('--dataset'   , help='dataset'           , type=str     , default='mitbih_arr')
    parser.add_argument('--seed'      , help='random seed'       , type=int     , default=6206)
    parser.add_argument('--save'      , help='save exp result?'  , type=str2bool, default=True)
    parser.add_argument('--n_shots'   , help='n per class data', type=int     , default=5)
    # kNN Arguments
    parser.add_argument('--decimal'   , help='decimal point'     , type=int     , default=4)
    parser.add_argument('--k'         , help='top-k for knn'     , type=int     , default=3)
    parser.add_argument('--method'    , help='method'            , type=str     , default='str', choices=['default', 'fpq', 'hybrid', 'cw', 'all'])
    # Neural Network Trainer Arguments
    parser.add_argument('--model'     , help='model'             , type=str     , default='CNN')
    parser.add_argument('--epochs'    , help='training epoch'    , type=int     , default=50)
    parser.add_argument('--batch_size', help='batch size'        , type=int     , default=120)
    parser.add_argument('--use_tb'    , help='use tensorboard'   , type=str2bool, default=False)
    args = parser.parse_args()
    return args

def check_path(path):
    os.makedirs(path, exist_ok=True)

def save_result(args, y_true, y_pred, save_path='./exp_results'):
    check_path(save_path)
    info = {'exp_name': args.exp_name,
            'seed': args.seed,
            'decimal': args.decimal,
            'n_shots': args.n_shots,
            'method': args.method,
            'y_true': y_true,
            'y_pred': y_pred}
    with open(f'{save_path}/{args.exp_name}_{args.decimal}_{args.n_shots}_{args.k}_{args.method}.pickle', 'wb') as f:
        pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)

def load_result(args, save_path='./exp_results'):
    with open(f'{save_path}/{args.exp_name}_{args.decimal}_{args.n_shots}_{args.k}_{args.method}.pickle', 'rb') as f:
        info = pickle.load(f)
    #print(info)
    
def class_repr(class_idx):
    return ["LBBB", 'N', 'PB', 'PVC','RBBB','VF'][int(class_idx)]


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

def build_model(args):
    input_size        = cfg.dataset_cfg[args.dataset]['input_size']
    n_channel         = cfg.dataset_cfg[args.dataset]['n_channel']
    n_class           = cfg.N_CLASS[args.dataset]
    return getattr(models, args.model)(input_size, input_channel=n_channel, num_label=n_class)

def convert_array(array, specific_numbers):
    """
    주어진 NumPy 배열에서 특정 숫자는 1로, 나머지는 0으로 변환합니다.

    Args:
    array (np.ndarray): 변환할 NumPy 배열.
    specific_numbers (list or set): 1로 바꿀 특정 숫자들의 목록.

    Returns:
    np.ndarray: 변환된 NumPy 배열.
    """
    # 주어진 배열과 특정 숫자를 비교하여 True/False로 구성된 마스크를 생성
    mask = np.isin(array, specific_numbers)
    
    # 마스크를 이용하여 배열의 값을 변환
    converted_array = np.where(mask, 1, 0)
    
    return converted_array