
import os
import sys
sys.path.append(os.path.abspath('..'))
from models.MTSC.ResNet import ResNet as ResNetM
from models.UTSC.ResNet import ResNet as ResNetU
from models.MTSC.FCN import FCN as FCNM
from models.UTSC.FCN import FCN as FCNU
from models.MTSC.InceptionTime import InceptionTime
import torch
import torch.optim as optim
def get_model(choose, channels, nb_classes, depth = 5, verify = False):
    depth = 5
    if choose == "ResNet":
        if channels > 1:
            model = ResNetM(channels, nb_classes, depth, [], True, False, verify)
        else:
            model = ResNetU(channels, nb_classes, depth, [], True, False, verify)
    elif choose == "InceptionTime":
        model = InceptionTime(channels, nb_classes, depth, [], True, False)
    elif choose == "FCN":
        if channels > 1:
            model = FCNM(channels, nb_classes, depth, [], True, False)
        else:
            model = FCNU(channels, nb_classes, depth, [], True, False)
    return model

def regulate_model(choose, channels, nb_classes, skip, verify = False):
    depth = 5
    if choose == "ResNet":
        if channels > 1:
            model = ResNetM(channels, nb_classes, depth, skip, False, verify)
        else:
            model = ResNetU(channels, nb_classes, depth, skip, False, verify)
    elif choose == "InceptionTime":
        model = InceptionTime(channels, nb_classes, depth, skip, False)
    elif choose == "FCN":
        if channels > 1:
            model = FCNM(channels, nb_classes, depth, skip, False)
        else:
            model = FCNU(channels, nb_classes, depth, skip, False)
    return model



def get_optimizer(lr, weight_decay, model, verify = 0):
    if verify == 1:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=1, patience=50) 
    return [optimizer,schedular]







    