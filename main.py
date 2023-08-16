import os
import sys
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
import torch
import random
import numpy as np
from thop import profile
import argparse
from base.data_process import get_split_dataset
from base.base_model import get_model,get_optimizer
from regulator.train_model import train
RANDOM_SEED=894251
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) #  
    torch.backends.cudnn.deterministic = True #  
    torch.backends.cudnn.benchmark = False 



def parse_args():
    parser = argparse.ArgumentParser(description='TCE')
    parser.add_argument('--seed', default=42, type=int, help='random seed to set')
    parser.add_argument('--loader', default='UEA',type=str,  help='The data loader used to load the experimental data. This can be set to UCR or UEA.')
    parser.add_argument('--dataset', default='UWaveGestureLibrary', type=str, help='dataset to use (mtsc/utsc)')
    parser.add_argument('--data_root', default="./datasets/Multivariate2018_ts/Multivariate_ts/", type=str, help='dataset path')
    parser.add_argument('--regulator', default=True, type=bool, help='Whether equipped with framework')
    parser.add_argument('--model', default="ResNet", type=str, help='1D-CNNs. This can be set to ResNet, InceptionTime or FCN.')
    parser.add_argument('--epochs', default=1500, type=int, help='training epoch')
    parser.add_argument('--alpha_epoch', default=100, type=int, help='regulatory epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='data batch size') 
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='optimizer of weight_decay') 
    parser.add_argument('--device', type=str, default="0", help='The cuda used for training and inference (defaults to 0)')
    parser.add_argument('--save_model', type=str, default="./", help='Save the checkpoint of best model')
    parser.add_argument('--save_result', type=str, default='./result.csv', help='Print the accuracy, change of params and flops.') 
    parser.add_argument('--channels', default=1, type=int, help='The channels in data set')
    parser.add_argument('--classes', default=1, type=int, help='The kinds of labels in data set')
    parser.add_argument('--length', default=1, type=int, help='Time series length')
    parser.add_argument('--filter', type=str, choices = ["Low0","Low1", "Low2", "Low3", "Low4", "Low5", "LFCs", "HFCs"], default="Low0", help='Add 0 percent to 25 percent LFCs to HFCs, all HFCs and all LFCs.)')
    parser.add_argument('--skip', type=int, choices = [2, 3, 4, 5], default=0, help='Choose skip layer.)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    print("Dataset:", args.dataset)
    print("Model:", args.model)
    print("Regulatory:", args.regulator)
    if args.device == "cpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    resultdf = pd.DataFrame()
    if args.loader == "UCR":
        args.data_root = "./datasets/Univariate2018_arff/Univariate_arff/"
    if args.loader == "HAR":
        args.data_root = "./datasets/HAR/"
    train_loader, val_loader, nb_classes, channels, length = get_split_dataset(args.loader, args.dataset, os.getcwd() + "/TCE/"+ args.data_root, args.batch_size, args.filter)
    args.channels = channels
    args.classes = nb_classes
    args.length = length
    model = get_model(args.model, channels, nb_classes).to(device)
    flops1, params1 = profile(model, (torch.randn((1,channels,length)).to(device),))
    optim = get_optimizer(args.lr, args.weight_decay, model)
    accuracy, flops2, params2 = train (model, optim, train_loader, val_loader, args)
    resultdf=resultdf.append([[args.model,args.regulator,args.dataset, accuracy, flops2, params2, (flops2 - flops1) / flops1, (params2 - params1) / params1]], ignore_index=True)
    resultdf.columns = ["model", "regulator","dataset", "acc", "flops", "params", "delta_flops", "delta_params"]
    resultdf.to_csv(args.save_result)
    print(resultdf)



    
