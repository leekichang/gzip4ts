__all__ = ['GzipClassifier']

from copy import deepcopy
import gc
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
from memory_profiler import profile

from utils import ExpLogger
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
        elif method in ['cw', 'all']:
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
    
    def from_XY(self, X, Y):
        self.X     = [self._GzipData().from_raw(X)]
        self.Y     = Y
        return self
    
    def precalculate(self):
        with ProcessPoolExecutor(max_workers=cpu_count()-4) as executor:
            futs = {}
            for i, x in enumerate(self.X):
                futs[executor.submit(x.precalculate)] = i
            
            for future in as_completed(futs):
                self.X[futs[future]] = future.result()

class GzipClassifier(object):
    # @profile
    def init(self, args):
#         print(
# f"""
# exp_name:{args.exp_name}
# dataset:{args.dataset}
# n_shots:{args.n_shots}
# decimal:{args.decimal}
# k:{args.k}
# method:{args.method}
# benchmark:{args.benchmark}
# seed:{args.seed}
# """)
        
        GzipCfg.decimal = args.decimal
        GzipCfg.k       = args.k
        GzipCfg.method  = args.method
        self.benchmark = args.benchmark
        
        if not args.benchmark:
            GzipCfg.seed    = args.seed
            GzipCfg.dataset = args.dataset
            GzipCfg.n_shots = args.n_shots
            
            self.distances_file = f"../dataset/{args.dataset}_{GzipCfg.decimal}_{GzipCfg.method}_{GzipCfg.n_shots}_{args.seed}_distances.npy"
            if os.path.isfile(self.distances_file):
                self.distances = np.load(self.distances_file)
        
            dataset_repr = GzipCfg.repr()
            dataset_file = f"../dataset/{dataset_repr}.npz"
            # is_reading_saved = not args.benchmark and 
            
            if os.path.isfile(dataset_file):
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
                trainset = datautils.load_raw_trainset_and_select_n_shots_per_class(args.dataset, GzipCfg.n_shots, "benchmark" if args.benchmark else None)
                testset = datautils.load_raw_testset(args.dataset)
                trainset = GzipDataManager().from_datamanager(trainset)
                testset  = GzipDataManager().from_datamanager(testset)
            
                # Precalculate
                trainset.precalculate()
                testset.precalculate()
                # Save the precalculated dataset
                np.savez_compressed(dataset_file, trainset=trainset, testset=testset)
        else:
            trainset = datautils.load_raw_trainset_and_select_n_shots_per_class(args.dataset, args.n_shots, "benchmark" if args.benchmark else None)
            testset = datautils.load_raw_testset(args.dataset)
            trainset_X = deepcopy(trainset.X[0].tolist())
            trainset_Y = deepcopy(trainset.Y[0].tolist())
            del trainset.X
            del trainset.Y
            del trainset
            # gc.collect()
            trainset = [(trainset_X, trainset_Y)]
            
            testset = deepcopy(testset[0][0])
        
        self.channelwise = GzipCfg.method in ['cw', 'all']
        self.hybrid      = GzipCfg.method in ['hybrid', 'all']
        
        return trainset, testset

    # @profile
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
            
            if not self.benchmark:
                x1 = testdata.get_rounded()
                Cx1 = (testdata.get_compressed_length())
                x2 = traindata.get_rounded()
                Cx2 = (traindata.get_compressed_length())
            else:
                x1 = np.round(testdata, GzipCfg.decimal)
                x2 = np.round(traindata, GzipCfg.decimal)
                if self.channelwise:
                    Cx1 = [len(gzip.compress(x1[i].tobytes())) for i in range(x1.shape[0])]
                    Cx2 = [len(gzip.compress(x2[i].tobytes())) for i in range(x2.shape[0])]
                else:
                    Cx1 = len(gzip.compress(x1.tobytes()))
                    Cx2 = len(gzip.compress(x2.tobytes()))
            
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

    # @profile
    def per_trainset(self, trainset, testset):
        distance_lists = [0]*len(trainset)
        if not self.benchmark:
            futs = {}
            with ThreadPoolExecutor(max_workers=32) as executor:
                for trainset_i, trainset in enumerate(trainset):
                    trainset, _ = trainset
                    fut = executor.submit(self.gzip_operation, trainset, testset)
                    futs[fut] = trainset_i
                
                for future in as_completed(futs):
                    distance_lists[futs[future]] = future.result()
        else:
            for trainset_i, trainset in enumerate(trainset):
                trainset, _ = trainset
                distance_lists[trainset_i] = self.gzip_operation(trainset, testset)
        return distance_lists
    
    
    # @profile
    def run(self, trainset, testset):
        if not self.benchmark:
            if self.distances is None:
                distance_lists = [0]*len(testset)
                # del self.testset
                with tqdm(total=len(testset) * len(trainset)) as pbar:
                    futs = {}
                    self.trainset = trainset
                    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                        for testset_i, _testset in enumerate(testset):
                            _testset, _ = _testset
                            
                            fut = executor.submit(self.per_trainset, trainset, _testset)
                            fut.add_done_callback(lambda x: pbar.update(1 * len(trainset)))
                            futs[fut] = testset_i
                            
                        for future in as_completed(futs):
                            distance_lists[futs[future]] = future.result()
                Y = testset.Y.tolist()
                np.save(self.distances_file, distance_lists)
            else:
                distance_lists = self.distances
                # self_testset = self.testset
                Y = testset.Y.tolist()
        else:
            distance_lists = [[self.gzip_operation(trainset[0][0], testset)]]
            Y = [0]
        
        pred = self.run_knn(trainset, distance_lists)
        return Y, pred
    
    def run_knn(self, trainset, distance_lists):
        pred = []
        if not self.benchmark:
            Y = trainset.Y
        else:
            Y = np.array([trainset[0][1]])
        for distance_from_x1 in distance_lists:
            # Per each test data, find the top K nearest neighbors
            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_class = Y[sorted_idx[:GzipCfg.k]].tolist()
            predict_class = max(set(top_k_class), key=top_k_class.count)
            pred.append(int(predict_class))
        return pred