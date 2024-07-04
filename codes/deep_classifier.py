import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import sklearn.metrics as metrics
from torch.utils.data import DataLoader

import utils
import codes.DataConfig as cfg
from models import *
from DataManager import TensorDataset

import torch.utils.tensorboard as tb

class SupervisedTrainer:
    def __init__(self, args):
        self.args         = args
        self.save_path    = f'./checkpoints/{args.exp_name}'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.trainset,\
        self.testset      = utils.load_dataset(dataset=args.dataset)
        self.trainset     = utils.choose_trainset(self.trainset, args)
        
        self.trainset     = TensorDataset(self.trainset)
        self.testset      = TensorDataset(self.testset)
        
        self.epoch        = 0
        self.epochs       = args.epochs
        self.device       = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True , drop_last=False)
        self.test_loader  = DataLoader(self.testset , batch_size=args.batch_size, shuffle=False, drop_last=False)
        self.model        = utils.build_model(args)# ResNet(input_size=input_size, input_channel=n_channel, num_label=n_class)
        
        self.model.to(self.device)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion    = nn.CrossEntropyLoss()
        
        self.train_loss = None
        self.test_loss  = None
        
        self.TB_WRITER = tb.SummaryWriter(f'./tensorboard/{str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))}_{self.args.exp_name}') \
            if self.args.use_tb else None        

    def train(self):
        self.model.train()
        losses = []
        for X, Y in self.train_loader:
            self.optimizer.zero_grad()
            X = X.to(self.device)
            Y = Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
        self.train_loss = np.mean(losses)
        
        if self.args.use_tb:
            self.TB_WRITER.add_scalar("Train Loss", self.train_loss, self.epoch+1)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        preds, targets, losses = [], [], []
        for X, Y in self.test_loader:
            X = X.to(self.device)
            Y = Y.to(self.device)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            preds.append(np.argmax(pred.cpu().numpy(), axis=-1))
            targets.append(Y.cpu().numpy())
            losses.append(loss.item())
            
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        self.acc = metrics.accuracy_score(y_true=targets, y_pred=preds)
        self.bacc = metrics.balanced_accuracy_score(y_true=targets, y_pred=preds)
        self.test_loss = np.mean(losses)
        if self.args.use_tb:
            self.TB_WRITER.add_scalar(f'Test Loss', self.test_loss, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Test Accuracy', self.acc, self.epoch+1)
            self.TB_WRITER.add_scalar(f'Test Accuracy (Balanced)', self.bacc, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Sensitivity', sens, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'F1-Score', f1, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'Specificity', spec, self.epoch+1)
        #     self.TB_WRITER.add_scalar(f'AUROC', auroc, self.epoch+1)
    
    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/{self.epoch+1}.pth')

    def print_train_info(self):
        print(f'({self.epoch+1:03}/{self.epochs}) Train Loss:{self.train_loss:>6.4f} Test Loss:{self.test_loss:>6.4f} ACC:{self.acc:.4f}  BalACC:{self.bacc:.4f}', flush=True)

    def save_n_shot(self):
        if not os.path.isfile(f'./results/n_shot_{self.args.dataset}.csv'):
            result = {self.args.n_shots:{self.args.model:self.bacc}}
            df = pd.DataFrame.from_dict(result)
        else:
            df = pd.read_csv(f'./results/n_shot_{self.args.dataset}.csv', encoding='cp949', index_col=0)

            # Check if the model exists in the DataFrame
            if self.args.model in df.index:
                # Check if the n_shots exists for the model
                if str(self.args.n_shots) in df.columns:
                    df.loc[self.args.model, str(self.args.n_shots)] = self.bacc
                else:
                    df[str(self.args.n_shots)] = None  # Add a new column for the n_shots
                    df.loc[self.args.model, str(self.args.n_shots)] = self.bacc
            else:
                # Add the model if it doesn't exist
                df.loc[self.args.model] = None
                df[str(self.args.n_shots)] = None
                df.loc[self.args.model, str(self.args.n_shots)] = self.bacc

            # Rename the index column to the model name
        df.index.name = 'Model'
        print(df)
        df.to_csv(f'./results/n_shot_{self.args.dataset}.csv', encoding='cp949')

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    torch.manual_seed(args.seed)
    trainer = SupervisedTrainer(args)
    for epoch in tqdm(range(trainer.epochs)):
        trainer.train()
        trainer.test()
        trainer.print_train_info()
        if (trainer.epoch+1)%10 == 0 and args.save:
            trainer.save_model()
        trainer.epoch += 1
        
    trainer.save_n_shot()