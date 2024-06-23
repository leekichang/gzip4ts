import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader

import utils
import config as cfg
from models import *
from DataManager import TensorDataset
from time import time

import torch.utils.tensorboard as tb

class SupervisedTrainer:
    def __init__(self, args):
        self.args         = args
        
        self.trainset,\
        self.testset      = utils.load_dataset(dataset=args.dataset, generate=True)
        self.testset.X = self.testset.X[:5]
        self.testset.Y = self.testset.Y[:5]
        self.testset      = TensorDataset(self.testset)
        
        self.epoch        = 0
        self.device       = 'cpu'

        self.test_loader  = DataLoader(self.testset , batch_size=1, shuffle=False, drop_last=False)
        self.model        = utils.build_model(args)# ResNet(input_size=input_size, input_channel=n_channel, num_label=n_class)
        
        self.model.to(self.device)
        
        self.test_loss  = None   

    @torch.no_grad()
    def test(self):
        self.model.eval()
        
        total_time = 0
        
        for i, (X, Y) in enumerate(self.test_loader):
            start_time = time()
            # X = X.reshape(X.shape[0], -1)
            X = X.to(self.device)
            self.model(X)
            if i > 0:
                total_time += time() - start_time
        
        return (total_time / (len(self.test_loader) - 1))

if __name__ == '__main__':
    args = utils.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    trainer = SupervisedTrainer(args)
    
    total_time = 0
    n_run = 5
    for i in range(n_run):
        start_time = time()
        t = trainer.test()
        total_time += t
    print(total_time / (n_run))
    