import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import utils
import config as cfg
from models import *
from DataManager import TensorDataset

import torch.utils.tensorboard as tb

class MLTrainer:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        self.trainset,\
        self.testset      = utils.load_dataset(dataset=args.dataset)
        self.trainset     = utils.choose_trainset(self.trainset, args)
        self.model        = AdaBoostClassifier() if args.model == 'AB' else RandomForestClassifier()

    def train(self):
        B = self.trainset.X.shape[0]
        data = self.trainset.X.reshape(B, -1)
        self.model.fit(X=data, y=self.trainset.Y)

    def test(self):
        B = self.testset.X.shape[0]
        data = self.testset.X.reshape(B, -1)
        targets = self.testset.Y
        preds = self.model.predict(X=data)
        self.acc = metrics.accuracy_score(y_true=targets, y_pred=preds)
        self.bacc = metrics.balanced_accuracy_score(y_true=targets, y_pred=preds)
        print(f'ACC:{self.acc:.4f}  BalACC:{self.bacc:.4f}', flush=True)


    # def train(self):
    #     self.model.train()
    #     losses = []
    #     for X, Y in self.train_loader:
    #         self.optimizer.zero_grad()
    #         X = X.to(self.device)
    #         Y = Y.to(self.device)
    #         pred = self.model(X)
    #         loss = self.criterion(pred, Y)
    #         losses.append(loss.item())
    #         loss.backward()
    #         self.optimizer.step()
    #     self.train_loss = np.mean(losses)
        
    #     if self.args.use_tb:
    #         self.TB_WRITER.add_scalar("Train Loss", self.train_loss, self.epoch+1)
        
    # @torch.no_grad()
    # def test(self):
    #     self.model.eval()
    #     preds, targets, losses = [], [], []
    #     for X, Y in self.test_loader:
    #         X = X.to(self.device)
    #         Y = Y.to(self.device)
    #         pred = self.model(X)
    #         loss = self.criterion(pred, Y)
    #         preds.append(np.argmax(pred.cpu().numpy(), axis=-1))
    #         targets.append(Y.cpu().numpy())
    #         losses.append(loss.item())
            
    #     preds = np.concatenate(preds)
    #     targets = np.concatenate(targets)
    #     self.acc = metrics.accuracy_score(y_true=targets, y_pred=preds)
    #     self.bacc = metrics.balanced_accuracy_score(y_true=targets, y_pred=preds)
    #     self.test_loss = np.mean(losses)
    #     if self.args.use_tb:
    #         self.TB_WRITER.add_scalar(f'Test Loss', self.test_loss, self.epoch+1)
    #         self.TB_WRITER.add_scalar(f'Test Accuracy', self.acc, self.epoch+1)
    #         self.TB_WRITER.add_scalar(f'Test Accuracy (Balanced)', self.bacc, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Sensitivity', sens, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'F1-Score', f1, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Specificity', spec, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'AUROC', auroc, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    trainer = MLTrainer(args)
    trainer.train()
    trainer.test()
    
    # for epoch in tqdm(range(trainer.epochs)):
    #     trainer.train()
    #     trainer.test()
    #     trainer.print_train_info()
    #     if (trainer.epoch+1)%10 == 0 and args.save:
    #         trainer.save_model()
    #     trainer.epoch += 1