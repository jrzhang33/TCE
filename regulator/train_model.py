import os
import sys
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data
import torch
import numpy as np
from base.Test import test
from thop import profile
from datetime import *
import math
from .Regulator_Framework import Regulator
from base.base_model import regulate_model, get_model
def train (model, optims, train_loader, val_loader, args):
    optimizer = optims[0]
    scheduler = optims[1]
    save_path = args.save_model + args.dataset + "_" + args.model +"_" + str(args.regulator) +".pth"
    train_steps = len(train_loader)
    loss_function = torch.nn.CrossEntropyLoss()
    if args.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_loss = math.inf
    flops2, params2 = profile(model, (torch.randn((1, args.channels, args.length)).to(device),))
    for epoch in range(args.epochs):
        if args.regulator and epoch == args.alpha_epoch:
            print(">>>>>>>>>>Regulating....")
            skip, focus = Regulator(model)
            if args.skip:
                skip = [args.skip - 1]
            print(skip)
            if skip:
                print("Updata !<<<<<<<<<<<<")
                best_loss = math.inf
                model = regulate_model(args.model, args.channels, args.classes, skip).to(device)
                new_state_dict = torch.load(save_path)
                pretrained_dict = {k: v for k, v in new_state_dict.items() if k.find('fc')==-1}
                model.load_state_dict(pretrained_dict,strict=False)
                flops2, params2 = profile(model, (torch.randn((1, args.channels, args.length)).to(device),))
            optimizer = optim.Adam (model.parameters(), lr=0.0001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, verbose=1, patience=100)  
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        model.train()
        for step, data in enumerate(train_bar):
            x_train1, x_train2, labels, y_train1, y_train2= data
            optimizer.zero_grad()
            try:
                outputs = model(x_train1)
            except:
                outputs = model(x_train1.squeeze(3))
            if len(outputs.shape)==1:
                outputs=torch.unsqueeze(outputs,0)        
            loss = loss_function(outputs, y_train1.long())
            loss.backward()
            optimizer.step()           
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                    args.epochs,
                                                                    loss)
        if  running_loss / train_steps < best_loss:
            best_loss =  running_loss / train_steps
            torch.save(model.state_dict(), save_path)
            es=0
        else:
            es += 1
            if es > 50 and args.regulator == False:
                break
            elif es > 500 and args.regulator:
                break
        scheduler.step(running_loss)
        if epoch % 100 == 0:
            accuracy, loss = test (val_loader, model, args.classes, len(val_loader))
            print(accuracy,flops2)


    # Validate
    try:
        model.load_state_dict(torch.load(save_path),strict=False)
    except:
        model = get_model(args.model, args.channels, args.classes).to(device)
        model.load_state_dict(torch.load(save_path),strict=False)
    accuracy, loss = test (val_loader, model, args.classes, len(val_loader))
    return accuracy, flops2, params2