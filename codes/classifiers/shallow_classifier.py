from copy import deepcopy
import os
import pickle
import numpy as np
import pandas as pd
import gc
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessClassifier
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from data.types import DataManager
import data.utils as datautils
import joblib

from memory_profiler import profile
class MLTrainer:
    # @profile
    def init(self, args):
        self.model   = args.model
        self.n_shots = args.n_shots
        self.benchmark = args.benchmark
        
        testset  = datautils.load_raw_testset(args.dataset)
        if self.benchmark:
            testset = DataManager(testset.X[:1].copy(), testset.Y[:1].copy())
            trainset = None
        else:
            testset = DataManager(testset.X, testset.Y)
            trainset = datautils.load_raw_trainset_and_select_n_shots_per_class(args.dataset, self.n_shots)
        
        self.model        = \
            SVC() if self.model == 'SVC' else \
            DecisionTreeClassifier() if self.model == 'DT' else \
            RandomForestClassifier() if self.model == 'RF' else \
            GaussianProcessClassifier() if self.model == 'GP' else \
            AdaBoostClassifier() if self.model == 'AB' else \
            MLPClassifier()   if self.model == 'MLP' else None
            
        self.save_path = f"../model_ckpts/shallow/{args.dataset}_{args.n_shots}_{args.model}_{args.seed}.pkl"
        
        return trainset, testset
    
    def train(self, trainset, load_saved=False):
        if load_saved and os.path.isfile(self.save_path):
            try:
                with open(self.save_path, "rb") as f:
                    self.model = pickle.load(f)
            except:
                self.model = joblib.load(self.save_path)
                with open(self.save_path, "wb") as f:
                    pickle.dump(self.model, f)
                with open(self.save_path, "rb") as f:
                    self.model = pickle.load(f)
        else:
            trainset = DataManager(trainset.X, trainset.Y)
            
            B = trainset.X.shape[0]
            data = trainset.X.reshape(B, -1)
            self.model.fit(X=data, y=trainset.Y)
            with open(self.save_path, "wb") as f:
                pickle.dump(self.model, f)

        
    def test(self, testset):
        B = testset.X.shape[0]
        data = testset.X.reshape(B, -1)
        targets = testset.Y
        preds = self.model.predict(X=data)
        return targets, preds