from copy import deepcopy
import os
import gzip
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from statistics import *

from time import time
import utils
import config as cfg
from DataManager import DataManager
import warnings
warnings.filterwarnings("ignore")

# def run_quant_fp_inner(x2, n_channel, Cx1, x1):
#     x2 = np.round(x2, dec_point)
#     distance_per_channel = 0
#     for i in range(n_channel):
#         Cx1_c = Cx1[i]
#         x1_c = x1[i]
        
#         x2_c = x2[i]
#         Cx2_c = len(gzip.compress(x2_c.tobytes()))
        
#         x1x2_c = np.concatenate([x1_c, x2_c])
#         Cx1x2_c = len(gzip.compress(x1x2_c.tobytes()))
        
#         ncd = (Cx1x2_c - min(Cx1_c, Cx2_c)) / max(Cx1_c, Cx2_c)

#         mse = np.linalg.norm(x1_c - x2_c, ord=2) # / len(x1_c)
#         # mse = np.log(mse)
#         # print(ncd, mse)
#         distance = harmonic_mean([ncd, mse]) #(ncd * mse)
#         # distance = ncd
#         distance_per_channel += distance
#     return distance_per_channel

def run_quant_fp_inner(x1x2_c, Cx1_c):
    try:
        st = time()
        x2_c = x1x2_c[14400:]
        Cx2_c = len(gzip.compress(x2_c))
        
        # x1x2_c = np.concatenate([x1_c, x2_c])
        # print("1", time() - st)
        Cx1x2_c = len(gzip.compress(x1x2_c))
        
        ncd = (Cx1x2_c - min(Cx1_c, Cx2_c)) / max(Cx1_c, Cx2_c)

        # mse = np.linalg.norm(x1_c - x2_c, ord=2) # 시간 측정위해서 원래 있어야하는데 빼버림
        distance = harmonic_mean([ncd, 0]) #(ncd * mse)
        # print("3", time() - st)
        return distance
    except:
        import traceback
        print(traceback.format_exc())
    
def run_quant_fp_inner2(x1_perc):
    return len(gzip.compress(x1_perc))

def run_quant_fp(exec):
    # x1 = np.round(x1, dec_point)
    n_channel = trainset.X.shape[1]
    
    x1 = [trainset.X[0, c][:14400] for c in range(n_channel)]
    
    inner1_futs = {}
    Cx1 = [None] * n_channel
    for i, x1_perc in enumerate(x1):
        inner1_futs[exec.submit(run_quant_fp_inner2, x1_perc)] = i
    
    for fut in as_completed(inner1_futs):
        i = inner1_futs[fut]
        Cx1[i] = fut.result()
    
    # distance_from_x1 = []
    # for i, (x2, _) in enumerate(self.trainset):
    #     #         futures[exec.submit(self.run_quant_fp_inner, x2, n_channel, Cx1, x1)] = i
    #     distance_from_x1.append(self.run_quant_fp_inner(x2, n_channel, Cx1, x1))
    distance_from_x1 = [0] * len(trainset)
    futures = {}
    for i, (x1x2, _) in enumerate(trainset):
        # x2 = np.round(x2, dec_point)
        for j in range(n_channel):
            x1x2_c = x1x2[j]
            Cx1_c = Cx1[j]
            futures[exec.submit(run_quant_fp_inner, x1x2_c, Cx1_c)] = (i, j)
    
    for fut in as_completed(futures):
        i, j = futures[fut]
        distance_from_x1[i] = fut.result()
        
    return distance_from_x1

def run():
    with ThreadPoolExecutor(max_workers=8) as exec:
        st = time()
        distance_lists = run_quant_fp(exec)
        for distance_from_x1 in distance_lists:
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = trainset.Y[sorted_idx[:K]].tolist()
            predict_class = max(set(top_k_class), key=top_k_class.count)
        t = time() - st
    return t


if __name__ == '__main__':
    # python temp_copy.py --exp_name quant_fp_1004_seed1 --seed 1 --n_shot 20 --k 5 --decimal 4 --dtype quant_fp --dataset mitbih_arr
    args = utils.parse_args()
    dec_point = args.decimal
    K         = args.k
    print(args)
    trainset, testset = utils.load_dataset(dataset=args.dataset, generate=True)
    trainset = utils.choose_trainset(trainset, args)
    
    testset.X = testset.X[:1]
    testset.Y = testset.Y[:1]
    
    trainset.X = np.array([[
        np.concatenate([np.round(testset.X[0, c], dec_point), np.round(trainset.X[i, c], dec_point)]).tobytes() \
            for c in range(trainset.X.shape[1])] for i in range(trainset.X.shape[0])])
    # testset.X = np.array([[.tobytes() for c in range(testset.X.shape[1])] for i in range(testset.X.shape[0])])
  
    _trainset = deepcopy(trainset)
    
    total_time = 0
    n_run = 6
    for i in range(n_run):
        trainset = deepcopy(_trainset)
        t = run()
        if i > 0:
            total_time += t
    print(total_time / (n_run - 1))
   