import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from time import time
import utils

import torch.utils.tensorboard as tb

class MLTrainer:
    def __init__(self, args):
        self.args         = args
        self.trainset,\
        self.testset      = utils.load_dataset(dataset=args.dataset, generate=True)
        self.trainset.X = self.trainset.X[:-1]
        self.trainset.Y = self.trainset.Y[:-1]
        self.trainset   = utils.choose_trainset(self.trainset, args)
        
        self.testset.X = self.testset.X[-1:]
        self.testset.Y = self.testset.Y[-1:]
        self.model        = AdaBoostClassifier() if args.model == 'AB' else RandomForestClassifier()

    def train(self):
        B = self.trainset.X.shape[0]
        data = self.trainset.X.reshape(B, -1)
        self.model.fit(X=data, y=self.trainset.Y)

    def test(self):
        B = self.testset.X.shape[0]
        data = self.testset.X.reshape(B, -1)
        preds = self.model.predict(X=data)


if __name__ == '__main__':
    args = utils.parse_args()
    print(args)
    trainer = MLTrainer(args)
    trainer.train()
    
    total_time = 0
    n_run = 6
    for i in range(n_run):
        start_time = time()
        trainer.test()
        if i > 0:
            t = time() - start_time
            total_time += t
    print(total_time / (n_run - 1))