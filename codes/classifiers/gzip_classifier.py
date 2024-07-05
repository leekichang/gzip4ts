__all__ = ['GzipClassifier']

import itertools
import os
import gzip
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as metrics
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from statistics import *

from utils import common
import data.utils as datautils
from data.types import DataManager

import time

class GzipCfg:
    dataset = None
    decimal = None
    method = None
    k = None
    n_shots = None
    seed = None

    @staticmethod
    def repr():
        method = GzipCfg.method
        if method in ['default']:
            method = "default" # no rounding, no channelwise compression
        elif method in ['fpq', 'hybrid']:
            method = "fpq" # rounding, no channelwise compression
        else:
            method = "cw" # all
        return f"{GzipCfg.dataset}_{GzipCfg.decimal}_{method}_{GzipCfg.n_shots}_{GzipCfg.seed}"

class GzipDataManager(DataManager):
    class _GzipData:
        def __init__(self):
            self.raw = None
            self.rounded = None
            self.compressed = None
        
        def from_raw(self, X):
            self.raw = X
            return self

        def calc_rounded(self):
            if GzipCfg.method in ['default']:
                self.rounded = self.raw
            else:
                self.rounded = np.round(self.raw, GzipCfg.decimal)
        
        def calc_compressed(self):
            if GzipCfg.method in ['all', 'cw']:
                self.compressed = [len(gzip.compress(self.rounded[i].tobytes())) for i in range(self.rounded.shape[0])]
            else:
                self.compressed = len(gzip.compress(self.rounded.tobytes()))

        def precalculate(self):
            self.calc_rounded()
            self.calc_compressed()
            
            return self
        
        def get_rounded(self):
            if self.rounded is None:
                self.calc_rounded()
            return self.rounded

        def get_compressed_length(self):
            if self.compressed is None:
                self.calc_compressed()
            return self.compressed
        
        @property
        def shape(self):
            return self.raw.shape
    
    def __init__(self):
        pass
    
    def from_saved(self, XYs):
        self.X = [xy[0] for xy in XYs]
        self.Y = np.array([int(xy[1]) for xy in XYs])
        return self
    
    def from_datamanager(self, datamanager: DataManager):
        self.X     = [self._GzipData().from_raw(X) for X in datamanager.X]
        self.Y     = datamanager.Y
        return self
    
    def precalculate(self):
        with ProcessPoolExecutor(max_workers=cpu_count()-4) as executor:
            futs = {}
            for i, x in enumerate(self.X):
                futs[executor.submit(x.precalculate)] = i
            
            for future in as_completed(futs):
                self.X[futs[future]] = future.result()

class GzipClassifier(object):
    def __init__(self, args):
        print(
f"""
exp_name:{args.exp_name}
dataset:{args.dataset}
n_shots:{args.n_shots}
decimal:{args.decimal}
k:{args.k}
method:{args.method}
benchmark:{args.benchmark}
seed:{args.seed}
""")
        GzipCfg.dataset = args.dataset
        GzipCfg.decimal = args.decimal
        GzipCfg.method  = args.method
        GzipCfg.n_shots = args.n_shots
        GzipCfg.k       = args.k
        GzipCfg.seed    = args.seed
        self.benchmark = args.benchmark
        
        dataset_repr = GzipCfg.repr()
        dataset_file = f"../dataset/{dataset_repr}.npz"
        is_reading_saved = not args.benchmark and os.path.isfile(dataset_file)
        
        if is_reading_saved:
            saved = np.load(dataset_file, allow_pickle=True)
            # trainset_x = saved['trainset_x'].tolist()
            # trainset_y = saved['trainset_y'].tolist()
            # trainset = DataManager(trainset_x, trainset_y)
            # testset_x = saved['testset_x'].tolist()
            # testset_y = saved['testset_y'].tolist()
            # testset = DataManager(testset_x, testset_y)
            trainset, testset = saved['trainset'].tolist(), saved['testset'].tolist()
            trainset = GzipDataManager().from_saved(trainset)
            testset  = GzipDataManager().from_saved(testset)
        else:
            trainset, testset = datautils.load_raw_dataset(dataset=GzipCfg.dataset)
            trainset = datautils.select_n_shots_per_class(trainset, GzipCfg.dataset, GzipCfg.n_shots)

            trainset = GzipDataManager().from_datamanager(trainset)
            testset  = GzipDataManager().from_datamanager(testset)
        
        if not args.benchmark and not is_reading_saved:
            # Precalculate
            trainset.precalculate()
            testset.precalculate()
            # Save the precalculated dataset
            np.savez_compressed(dataset_file, trainset=trainset, testset=testset)
        
        self.trainset   = trainset
        self.testset    = testset
        
        self.channelwise = GzipCfg.method in ['cw', 'all']
        self.hybrid      = GzipCfg.method in ['hybrid', 'all']

    def gzip_operation(
        self,
        traindata, testdata
        ):
        
        try:
            def calc(_x1, _x2, _Cx1, _Cx2):
                x1x2 = np.concatenate([_x1, _x2])
                Cx1x2 = len(gzip.compress(x1x2.tobytes()))
                ncd = (Cx1x2 - min(_Cx1, _Cx2)) / max(_Cx1, _Cx2)
                
                if self.hybrid:
                    xd = _x1 - _x2
                    # if self.channelwise:
                    #     mse = np.linalg.norm(xd, ord=2)
                    # else:
                    #     mse = sum([np.linalg.norm(xd[i], ord=2) for i in range(n_channel)]) 
                    
                    # for speed, use flatten
                    if len(xd.shape) > 1:
                        xd = xd.flatten()
                    mse = np.linalg.norm(xd, ord=2)
                    distance = harmonic_mean([ncd, mse])
                else:
                    distance = ncd
                return distance
            x1 = testdata.get_rounded()
            Cx1 = (testdata.get_compressed_length())
            x2 = traindata.get_rounded()
            Cx2 = (traindata.get_compressed_length())
            n_channel = x1.shape[0]
            
            if self.channelwise == True:
                distance = 0
                for i in range(n_channel):
                    distance += calc(x1[i], x2[i], Cx1[i], Cx2[i])
            else:
                distance = calc(x1, x2, Cx1, Cx2)
        except Exception as e:
            print(e)
            distance = np.inf
        return distance

    def per_trainset(self, testset):
        distance_lists = [0]*len(self.trainset)
        futs = {}
        with ThreadPoolExecutor(max_workers=32) as executor:
            for trainset_i, trainset in enumerate(self.trainset):
                trainset, _ = trainset
                fut = executor.submit(self.gzip_operation, trainset, testset)
                futs[fut] = trainset_i
            
            for future in as_completed(futs):
                distance_lists[futs[future]] = future.result()
        
        return distance_lists
    
    
    def run(self):
        distance_lists = [0]*len(self.testset)
        pred = []
        if not self.benchmark:
            self_testset = self.testset
            del self.testset
            with tqdm(total=len(self_testset) * len(self.trainset)) as pbar:
                futs = {}
                with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                    for testset_i, testset in enumerate(self_testset):
                        testset, _ = testset
                        
                        fut = executor.submit(self.per_trainset, testset)
                        fut.add_done_callback(lambda x: pbar.update(1 * len(self.trainset)))
                        futs[fut] = testset_i
                        
                    for future in as_completed(futs):
                        distance_lists[futs[future]] = future.result()
        else:
            raise NotImplementedError

        for distance_from_x1 in distance_lists:
            # Per each test data, find the top K nearest neighbors
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = self.trainset.Y[sorted_idx[:GzipCfg.k]].tolist()
            predict_class = max(set(top_k_class), key=top_k_class.count)
            pred.append(int(predict_class))
        return self_testset.Y.tolist(), pred