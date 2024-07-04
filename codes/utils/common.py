import os
import pickle

import numpy as np

import models
import data.config as cfg
from data.types import DataManager

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