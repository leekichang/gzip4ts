"""
Generates custom dataset MIT-BIH Auth
"""
import numpy as np

from data.utils import load_raw_dataset
trainset, testset = load_raw_dataset(
    dataset="mitbih_id",
)

authenticating_patient_id = 0
# for authenticating_patient_id in range(len(np.unique(trainset.Y))):

trainset_patient_id_smaller = trainset.Y < 20
train_x = trainset.X[trainset_patient_id_smaller]
train_y = trainset.Y[trainset_patient_id_smaller]

testset_patient_id_smaller = testset.Y < 20
test_x = testset.X[testset_patient_id_smaller]
test_y = testset.Y[testset_patient_id_smaller]

train_y = np.where(train_y == authenticating_patient_id, 1, 0)
test_y = np.where(test_y == authenticating_patient_id, 1, 0)


import os
dataset_dir = '../../dataset_kichang/mitbih_auth'
os.makedirs(dataset_dir, exist_ok=True)

np.save(f'{dataset_dir}/x_train.npy', train_x, allow_pickle=True)
np.save(f'{dataset_dir}/y_train.npy', train_y)
np.save(f'{dataset_dir}/x_test.npy', test_x)
np.save(f'{dataset_dir}/y_test.npy', test_y)