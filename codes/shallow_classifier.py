import os
import numpy as np
import pandas as pd
from datetime import datetime
import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

import utils
from codes.main import convert_array

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
        targets = convert_array(targets, 2)
        preds = convert_array(preds, 2)
        self.acc = metrics.accuracy_score(y_true=targets, y_pred=preds)
        self.bacc = metrics.balanced_accuracy_score(y_true=targets, y_pred=preds)
        if not os.path.isfile(f'./results/n_shot_{self.args.dataset}_{self.args.model}.csv'):
            result = {self.args.n_shots:{self.args.model:self.bacc}}
            df = pd.DataFrame.from_dict(result)
        else:
            df = pd.read_csv(f'./results/n_shot_{self.args.dataset}_{self.args.model}.csv', encoding='cp949', index_col=0)

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
        df.to_csv(f'./results/n_shot_{self.args.dataset}_{self.args.model}.csv', encoding='cp949')
        print(f'ACC:{self.acc:.4f}  BalACC:{self.bacc:.4f}', flush=True)

if __name__ == '__main__':
    from tqdm import tqdm
    args = utils.parse_args()
    trainer = MLTrainer(args)
    trainer.train()
    trainer.test()
    