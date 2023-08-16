
import os
import sys
sys.path.append(os.path.abspath('.'))
import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .MyDataset import MyDataset
from .Muldataloader import process_ts_data
from sktime.datasets import load_from_tsfile_to_dataframe
from .TSC_data_loader import TSC_multivariate_data_loader
from .ts_datasets import UCR_UEADataset
from regulator.TCE import Low2High
def readucr(filename, loader):
    if loader == "UCR":
        data = pd.read_csv(filename,sep="  ",header=None )
        Y = data.iloc[0:len(data),0]
        X = data.iloc[0:len(data),1:data.shape[1]]
        if X.shape[1] == 0:
            data = pd.read_csv(filename,sep=",",header=None )
            Y = data.iloc[0:len(data),0]
            X = data.iloc[0:len(data),1:data.shape[1]] 
            X[np.isnan(X)] = 0
        return X, Y
    else:
        data= load_from_tsfile_to_dataframe(filename)
        return data

def load_dataset_mul(dataset_path, dataset_name, normalize_timeseries=False, verbose=True) -> (np.array, np.array):
    if verbose: print("Loading train / test dataset : ", dataset_name)

    root_path = dataset_path + '/' + dataset_name + '/'
    x_train_path = root_path + "X_train.npy"
    y_train_path = root_path + "y_train.npy"
    x_test_path = root_path + "X_test.npy"
    y_test_path = root_path + "y_test.npy"

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path).astype(np.float32)
        y_train = np.squeeze(np.load(y_train_path))
        X_test = np.load(x_test_path).astype(np.float32)
        y_test = np.squeeze(np.load(y_test_path))
    else:
        raise FileNotFoundError('File %s not found!' % (dataset_name))

    is_timeseries = True

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished processing train dataset..")

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    if is_timeseries:
        # scale the values
        if normalize_timeseries:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, nb_classes

def get_split_dataset(loader, each, root, batch_size, l2h = "Low0"):
    if loader == "UEA":
        path, fname = root, each
        if each in ["InsectWingbeat","Phoneme"]:      
            X_train, y_train,X_test,y_test = TSC_multivariate_data_loader(path, fname)
        else:        
            X_train, y_train = readucr(path+fname+'/'+fname+'_TRAIN.ts', loader)
            X_train = process_ts_data(X_train, normalise=False)
            X_test, y_test = readucr(path+fname+'/'+fname+'_TEST.ts', loader)
            X_test = process_ts_data(X_test, normalise=False)
            class_le = LabelEncoder()
            y_train = class_le.fit_transform(y_train)
            y_test = class_le.fit_transform(y_test)
        if l2h !="Low0":
            for i in range(len(X_train)):
                for j in range(len(X_train[i])):
                    a,zeroo1=Low2High(torch.Tensor(X_train[i][j]).unsqueeze(0).unsqueeze(2),l2h)
                    X_train[i,j]=a.cpu().squeeze(0).numpy()      
            for i in range(len(X_test)):
                for j in range(len((X_test[i]))):
                    a,zeroo1=Low2High(torch.Tensor(X_test[i][j]).unsqueeze(0).unsqueeze(2),l2h)
                    X_test[i,j]=a.cpu().squeeze(0).numpy()

        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        channels=X_train.shape[1]
        batch_size = min(int(X_train.shape[0]/10), batch_size)  
        train_index=np.array(range(len(y_train))).reshape(len(y_train),1)
        val_index=np.array(range(len(y_test))).reshape(len(y_test),1)
        train_data=MyDataset(X_train,y_train,train_index)
        validation_data=MyDataset(X_test,y_test,val_index)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0,drop_last=True)

        validate_loader = torch.utils.data.DataLoader(validation_data,
                                            batch_size=batch_size, shuffle=True,num_workers=0,drop_last=True)
        return train_loader, validate_loader, nb_classes, channels, X_train.shape[-1]
    elif loader == 'UCR':
        path, fname = root, each
        try:
            x_train, y_train = readucr(path+fname+'/'+fname+'_TRAIN.txt', loader)
            x_train=x_train.to_numpy()
            y_train=y_train.to_numpy()
            x_test, y_test = readucr(path+fname+'/'+fname+'_TEST.txt', loader)
            x_test=x_test.to_numpy()
            y_test=y_test.to_numpy()
        except:
            x_train, y_train,x_test,y_test = [],[],[],[]
            train_dataset = UCR_UEADataset(fname, split="train",extract_path = "Univer")
            test_dataset = UCR_UEADataset(fname, split="test",extract_path = "Univer")
            for i in range(len(train_dataset)):
                x_train.append(train_dataset[i]['input'][:,0].numpy())
                y_train.append(train_dataset[i]['label'].numpy())
            for i in range(len(test_dataset)):
                x_test.append(test_dataset[i]['input'][:,0].numpy())
                y_test.append(test_dataset[i]['label'].numpy())
            x_train, y_train, x_test, y_test  = np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
        if l2h !="Low0": 
            for i in range(len(x_train)):
                a,zeroo1=Low2High(torch.Tensor(x_train[i]).unsqueeze(0).unsqueeze(2),l2h)
                x_train[i]=a.cpu().squeeze(0).numpy()    
            for i in range(len(x_test)):
                a,zeroo1=Low2High(torch.Tensor(x_test[i]).unsqueeze(0).unsqueeze(2),l2h)
                x_test[i]=a.cpu().squeeze(0).numpy()
        nb_classes = len(np.unique(y_test))
        y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
        y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
        channels=1
        batch_size = min(int(x_train.shape[0]/10), batch_size)
        batch_size_test = min(int(x_test.shape[0]/10), batch_size)
        x_train=x_train.reshape(x_train.shape[0],channels,x_train.shape[1],1)
        x_test=x_test.reshape(x_test.shape[0],channels,x_test.shape[1],1)
        train_index=np.array(range(len(y_train))).reshape(len(y_train),1)
        val_index=np.array(range(len(y_test))).reshape(len(y_test),1)
        train_data=MyDataset(x_train,y_train,train_index)
        validation_data=MyDataset(x_test,y_test,val_index)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0,drop_last=True)

        validate_loader = torch.utils.data.DataLoader(validation_data,
                                            batch_size=batch_size_test, shuffle=True,num_workers=0,drop_last=True)
        return train_loader, validate_loader, nb_classes, channels, x_train.shape[-2]
    else:
        X_train, y_train, X_test, y_test, nb_classes = load_dataset_mul(root, each)
        if l2h != "Low0" :
            for i in range(len(X_train)) :
                for j in range(len(X_train[i])) :
                    a, zeroo1 = Low2High(torch.Tensor(X_train[i][j]).unsqueeze(0).unsqueeze(2), l2h)
                    X_train[i, j] = a.cpu().squeeze(0).numpy()
            for i in range(len(X_test)) :
                for j in range(len((X_test[i]))) :
                    a, zeroo1 = Low2High(torch.Tensor(X_test[i][j]).unsqueeze(0).unsqueeze(2), l2h)
                    X_test[i, j] = a.cpu().squeeze(0).numpy()
        train_index = np.array(range(len(y_train))).reshape(len(y_train), 1)
        val_index = np.array(range(len(y_test))).reshape(len(y_test), 1)
        train_data = MyDataset(X_train, y_train, train_index)
        test_data = MyDataset(X_test, y_test, val_index)
        batch_size = min(len(y_train), 16)
        batch_size_test = min(len(y_test), 16)
        n_class = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        channels = X_train.shape[1]

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=0, drop_last=True)

        val_loader = torch.utils.data.DataLoader(test_data,
                                                 batch_size=batch_size_test, shuffle=shuffle,
                                                 num_workers=0, drop_last=True)
        return train_loader, val_loader, channels, n_class, X_train.shape[-1]