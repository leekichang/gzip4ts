import os
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as metrics
from multiprocessing import Pool
from statistics import *

import ops
import utils
import codes.DataConfig as cfg
from DataManager import DataManager

import time

class Knn(object):
    def __init__(self, trainset, args):
        self.trainset  = trainset
        self.dec_point = args.decimal
        self.K         = args.k

    def run_default(self, data):
        """
        Applied: none
        """
        x1, _ = data
        Cx1 = len(gzip.compress(x1.tobytes()))
        distance_from_x1 = []
        for x2, _ in self.trainset:
            Cx2 = len(gzip.compress(x2.tobytes()))
            x1x2 = np.concatenate([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.tobytes()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1

    def run_fpq(self, data):
        """
        Applied: floating point quantization
        """
        x1, _ = data
        x1 = np.round(x1, self.dec_point)
        Cx1 = len(gzip.compress(x1.tobytes()))
        distance_from_x1 = []
        for x2, _ in self.trainset:
            x2 = np.round(x2, self.dec_point)
            Cx2 = len(gzip.compress(x2.tobytes()))
            x1x2 = np.concatenate([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.tobytes()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance_from_x1.append(ncd)
        return distance_from_x1

    def run_hybrid(self, data):
        """
        Applied: floating point quantization, hybrid distance
        """
        x1, _ = data
        x1 = np.round(x1, self.dec_point)
        Cx1 = len(gzip.compress(x1.tobytes()))
        distance_from_x1 = []
        
        n_channel, seq_len = x1.shape
        
        for x2, _ in self.trainset:
            x2 = np.round(x2, self.dec_point)
            Cx2 = len(gzip.compress(x2.tobytes()))
            x1x2 = np.concatenate([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.tobytes()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            x1_c = x1
            x2_c = x2
            xd = x1_c - x2_c
            
            mse = sum([np.linalg.norm(xd[i], ord=2) for i in range(n_channel)])
            
            # mse = np.linalg.norm(xd, ord=2) # / len(x1_c)
            distance = harmonic_mean([ncd, mse]) #(ncd * mse)
            distance_from_x1.append(distance)
        return distance_from_x1
    
    
    def run_cw(self, data):
        """
        Applied: floating point quantization, channel-wise compression
        """
        x1, _ = data
        x1 = np.round(x1, self.dec_point)
        n_channel, seq_len = x1.shape
        Cx1 = [len(gzip.compress(x1[i].tobytes())) for i in range(n_channel)]
        distance_from_x1 = []
        for x2, _ in self.trainset:
            x2 = np.round(x2, self.dec_point)
            distance_per_channel = 0
            for i in range(n_channel):
                Cx1_c = Cx1[i]
                x1_c = x1[i]
                
                x2_c = x2[i]
                Cx2_c = len(gzip.compress(x2_c.tobytes()))
                
                x1x2_c = np.concatenate([x1_c, x2_c])
                Cx1x2_c = len(gzip.compress(x1x2_c.tobytes()))
                
                ncd = (Cx1x2_c - min(Cx1_c, Cx2_c)) / max(Cx1_c, Cx2_c)
                distance = ncd

                distance_per_channel += distance
            distance_from_x1.append(distance_per_channel)
        return distance_from_x1
    
    def run_all(self, data):
        """
        Applied: floating point quantization, channel-wise compression, hybrid distance
        """
        x1, _
        x1, _ = data
        x1 = np.round(x1, self.dec_point)
        n_channel, seq_len = x1.shape
        Cx1 = [len(gzip.compress(x1[i].tobytes())) for i in range(n_channel)]
        distance_from_x1 = []
        for x2, _ in self.trainset:
            x2 = np.round(x2, self.dec_point)
            distance_per_channel = 0
            for i in range(n_channel):
                Cx1_c = Cx1[i]
                x1_c = x1[i]
                
                x2_c = x2[i]
                Cx2_c = len(gzip.compress(x2_c.tobytes()))
                
                x1x2_c = np.concatenate([x1_c, x2_c])
                Cx1x2_c = len(gzip.compress(x1x2_c.tobytes()))
                
                ncd = (Cx1x2_c - min(Cx1_c, Cx2_c)) / max(Cx1_c, Cx2_c)

                xd = x1_c - x2_c
                mse = np.linalg.norm(xd, ord=2) # / len(x1_c)
                distance = harmonic_mean([ncd, mse]) #(ncd * mse)
                distance_per_channel += distance
            distance_from_x1.append(distance_per_channel)
        return distance_from_x1
    
    def run(self, testset, compress='all'):
        pred = []
        with Pool(processes=20) as pool:
            distance_lists = list(tqdm(pool.imap(getattr(self, f'run_{compress}'), testset), total=len(testset)))

        for distance_from_x1 in distance_lists:
            # Per each test data, find the top K nearest neighbors
            
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = self.trainset.Y[sorted_idx[:self.K]].tolist()
            predict_class = max(set(top_k_class), key=top_k_class.count)
            pred.append(predict_class)
        return testset.Y.tolist(), pred

def load_dataset(path="../dataset", dataset='mitbih_arr', sample=(-1, -1)):
    X = np.load(f'{path}/{dataset}/x_train.npy')
    Y = np.load(f'{path}/{dataset}/y_train.npy')
    trainset = DataManager(X, Y)
    X = np.load(f'{path}/{dataset}/x_test.npy')
    Y = np.load(f'{path}/{dataset}/y_test.npy')
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

def filter_testset(testset, args):
    np.random.seed(args.seed)
    X = []
    Y = np.array([])
    for i in range(cfg.N_CLASS[args.dataset]):
        idx = np.random.choice(len(testset.Y[testset.Y==i]), args.n_shots, replace=False)
        X.append(testset.X[testset.Y==i][idx])
        Y = np.hstack([Y, np.ones(args.n_shots)*i])
    X = np.concatenate(X, axis=0)
    testset = DataManager(X, Y)
    return testset


if __name__ == '__main__':
    args = utils.parse_args()
    print(f'exp_name:{args.exp_name}\ndecimal:{args.decimal}\nnum shot:{args.n_shots}\nmethod:{args.dtype}')
    trainset, testset = load_dataset(dataset=args.dataset)
    trainset = choose_trainset(trainset, args)
    testset  = testset
    print(args.dataset, "numclass:", max(trainset.Y)+1)
    
    
    
    knn = Knn(trainset=trainset, args=args)
    y_true, y_pred = knn.run(testset, compress=f'{args.dtype}')
    # y_true = convert_array(y_true, 1)
    # y_pred = convert_array(y_pred, 1)
    acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    bacc = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    conf_mat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    # print()
    # print(f'         ACC: { acc*100:>5.2f}%' , flush=True)
    # print(f'Balanced ACC: {bacc*100:>5.2f}%', flush=True)
    # print(conf_mat, flush=True)
    if not os.path.isfile(f'./results/n_shot_{args.dataset}_{args.model}.csv'):
        result = {args.n_shots:{args.model:bacc}}
        df = pd.DataFrame.from_dict(result)
    else:
        df = pd.read_csv(f'./results/n_shot_{args.dataset}_{args.model}.csv', encoding='cp949', index_col=0)
        # Check if the model exists in the DataFrame
        if args.model in df.index:
            # Check if the n_shots exists for the model
            if str(args.n_shots) in df.columns:
                df.loc[args.model, str(args.n_shots)] = bacc
            else:
                df[str(args.n_shots)] = None  # Add a new column for the n_shots
                df.loc[args.model, str(args.n_shots)] = bacc
        else:
            # Add the model if it doesn't exist
            df.loc[args.model] = None
            df[str(args.n_shots)] = None
            df.loc[sf.args.model, str(args.n_shots)] = bacc
        # Rename the index column to the model name
    df.index.name = 'Model'
    # print(df)
    df.to_csv(f'./results/n_shot_{args.dataset}_{args.model}.csv', encoding='cp949')
    print(f'ACC:{acc:.4f}  BalACC:{bacc:.4f}\n\n', flush=True)
    # if not os.path.isfile(f'./results/quant_{args.dataset}.csv'):
    #     result = {args.decimal:{args.model:bacc}}
    #     df = pd.DataFrame.from_dict(result)
    # else:
    #     df = pd.read_csv(f'./results/quant_{args.dataset}.csv', encoding='cp949', index_col=0)

    #     # Check if the model exists in the DataFrame
    #     if args.model in df.index:
    #         # Check if the n_shots exists for the model
    #         if str(args.decimal) in df.columns:
    #             df.loc[args.model, str(args.decimal)] = bacc
    #         else:
    #             df[str(args.decimal)] = None  # Add a new column for the n_shots
    #             df.loc[args.model, str(args.decimal)] = bacc
    #     else:
    #         # Add the model if it doesn't exist
    #         df.loc[args.model] = None
    #         df[str(args.decimal)] = None
    #         df.loc[args.model, str(args.decimal)] = bacc

    #     # Rename the index column to the model name
    # df.index.name = 'Model'
    # print(df)
    # df.to_csv(f'./results/quant_{args.dataset}.csv', encoding='cp949')
    # utils.save_result(args, y_true, y_pred)
    # utils.load_result(args)