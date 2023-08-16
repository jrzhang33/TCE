import os
import sys

sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
import torch.nn as nn
import datetime
from ts_datasets import UCR_UEADataset
import numpy as np


extract_path = "./datasets/Multivariate2018_ts"

now = datetime.datetime.now().strftime('%Y-%m-%d %H.%M')


# univariate_dataset = ["CricketX", "ECG200", "Wafer"]

# for dataset_name in multivariate_dataset[:2]:

def read_mul(dataset_name):
    train_dataset = UCR_UEADataset(dataset_name, split="train", extract_path=extract_path)
    test_dataset = UCR_UEADataset(dataset_name, split="test", extract_path=extract_path)
    X_train, y_train, X_test, y_test = [], [], [],[]
    for x in train_dataset:
        X_train.append(np.array(x['input']))
        y_train.append(np.array(x['label']))
    for x in test_dataset:
        X_test.append(np.array(x['input']))
        y_test.append(np.array(x['label']))
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
