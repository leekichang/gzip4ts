from copy import deepcopy
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

import classifiers.models
import data.config as cfg
from data.types import DataManager, TensorDataset

import torch.utils.tensorboard as tb
import data.utils as datautils

def build_model(args):
    input_size        = cfg.dataset_cfg[args.dataset]['input_size']
    n_channel         = cfg.dataset_cfg[args.dataset]['n_channel']
    n_class           = cfg.N_CLASS[args.dataset]
    return getattr(classifiers.models, args.model)(input_size, input_channel=n_channel, num_label=n_class)

class SupervisedTrainer:
    def init(self, args):
        
        testset  = datautils.load_raw_testset(args.dataset)
        self.benchmark = args.benchmark
        self.n_shots = args.n_shots
        
        if self.benchmark:
            testset = DataManager(testset.X[:1].copy(), testset.Y[:1].copy())
            testset      = TensorDataset(testset)
            trainset = None
        else:
            testset = DataManager(testset.X, testset.Y)
            trainset = datautils.load_raw_trainset_and_select_n_shots_per_class(args.dataset, self.n_shots)
            testset  = TensorDataset(testset)
            trainset = TensorDataset(trainset)
        
        self.epochs       = args.epochs
        try:
            self.device       = torch.device(args.device)
        except:
            self.device    = "cpu"
        self.model        = build_model(args) # ResNet(input_size=input_size, input_channel=n_channel, num_label=n_class)
        
        self.batch_size   = args.batch_size
        self.model.to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion    = nn.CrossEntropyLoss()
        
        model_repr = f"{args.dataset}_{args.n_shots}_{args.model}_{self.epochs}_{args.batch_size}_{args.seed}"
        self.save_path = f"../model_ckpts/deep/{model_repr}.pt"
        
        self.use_tb = args.use_tb and not self.benchmark
        # {str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}
        self.TB_WRITER = tb.SummaryWriter(f'../tensorboard/{model_repr}') \
            if self.use_tb else None        

        return trainset, testset
    
    def train_per_epoch(self, train_loader, epoch):
        self.model.train()
        losses = []
        preds, targets = [], []
        for X, Y in train_loader:
            self.optimizer.zero_grad()
            X = X.to(self.device)
            Y = Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            preds.append(np.argmax(pred.detach().cpu().numpy(), axis=-1))
            targets.append(Y.detach().cpu().numpy())
                
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        train_loss = np.mean(losses)
        
        if self.use_tb:
            acc = metrics.accuracy_score(y_true=targets, y_pred=preds)
            bacc = metrics.balanced_accuracy_score(y_true=targets, y_pred=preds)
            self.TB_WRITER.add_scalar(f'Train Accuracy', acc, epoch)
            self.TB_WRITER.add_scalar(f'Train Accuracy (Balanced)', bacc, epoch)
            self.TB_WRITER.add_scalar("Train Loss", train_loss, epoch)
        
    def train(self, trainset, load_saved=False):
        if load_saved and os.path.isfile(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path))
            return
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True , drop_last=False)
        for epoch in range(self.epochs):
            self.train_per_epoch(train_loader, epoch+1)
            # self.test()
            
        torch.save(self.model.state_dict(), self.save_path)
    
    def prepare_benckmark(self):
        del self.optimizer
        del self.criterion
    
    @torch.no_grad()
    def test(self, testset):
        self.model.eval()
        device = 'cpu' if self.benchmark else self.device
        batch_size = 1 if self.benchmark else self.batch_size
        
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        if self.benchmark:
            self.model.to('cpu')
        
        preds, targets = [], []
        for X, Y in testloader:
            X = X.to(device)
            pred = self.model(X)
            preds.append(np.argmax(pred.cpu().numpy(), axis=-1))
            targets.append(Y.numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        return targets, preds