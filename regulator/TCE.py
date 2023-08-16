import numpy as np
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.utils.data
import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
RANDOM_SEED=222222
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) #  
    torch.backends.cudnn.deterministic = True #  
    torch.backends.cudnn.benchmark = False 
set_seed(RANDOM_SEED)

#Obtain FE feature map
def transfft(x_input):
    x_fft_init = torch.fft.rfft(x_input[..., 0],dim=2,norm="forward")
    x_fft = torch.stack((x_fft_init.real, x_fft_init.imag), -1)
    x_fft = x_fft.detach().cpu().numpy()

    for i in range(x_fft.shape[0]): 
        if i==0:
            ff = np.sqrt(x_fft[i,:,:,0]**2 + x_fft[i,:,:,1]**2) 
            ff=ff.reshape(1,x_fft.shape[1],x_fft.shape[2],1)
            continue 
        f = np.sqrt(x_fft[i,:,:,0]**2 + x_fft[i,:,:,1]**2).reshape(1,x_fft.shape[1],x_fft.shape[2],1)
        ff=np.concatenate([ff,f],0)
    x_fft=torch.from_numpy(ff[:,:,:-1,:]).to(device)
    return x_fft,x_fft_init

#Calculate p for every FE feature map
def behavior(x:Tensor):
    x = x.unsqueeze(3)
    b, c, _ ,_ =  x.size()
    x_fft,x_fft_init = transfft(x)
    rms = nn.AdaptiveAvgPool2d(1)(x_fft[:,:,:]*x_fft[:,:,:]).sqrt().view(b,c,1,1)
    max=nn.AdaptiveMaxPool2d(1)(abs(x_fft[:,:,:])).view(b,c,1,1)
    y_p=max/rms 
    return y_p

def Low2High(x:Tensor,choose):
    x=torch.Tensor(x)
    x=x.unsqueeze(0)
    b, c, d , e= x.size()
    x_fft, x_fft_init=transfft(x)
    k1=torch.Tensor(np.arange(1,x_fft.shape[2]+1)).to(device).repeat(x_fft.shape[0],x_fft.shape[1],1)
    fc=k1.unsqueeze(3)*x_fft
    y_1=(torch.sum(fc,dim=2)/torch.sum(x_fft,dim=2)).view(b,c,1,1) 
    y_11 = nn.AdaptiveAvgPool2d(1)(x_fft[:,:,:]).view(b,c,1,1)
    y_22=nn.AdaptiveMaxPool2d(1)(abs(x_fft[:,:,:])).view(b,c,1,1)
    y_2=y_22/y_11 
    fc1=int(y_1)
    zeroo=[]
    
    if choose=="LFCs":  
        x_= x_fft_init
        x_[:,:,fc1:]=0
        x2=torch.fft.irfft(x_,n=d,dim=2)

    elif choose=="HFCs":
        x_= x_fft_init
        x_[:,:,:fc1]=0

        x2=torch.fft.irfft(x_,n=d,dim=2)
    elif choose=="Low0":
        x2=x

    elif choose=="Low1":  
        x_= x_fft_init
        for i in range(0, int(fc1 * 0.95)):
            x_[:,:,i] = 0
            zeroo.append(i)
        x2=torch.fft.irfft(x_,n=d,dim=2)

    elif choose=="Low2":
        x_= x_fft_init
        for i in range(0, int(fc1 * 0.9)):
            x_[:,:,i]=0
            zeroo.append(i)
        x2=torch.fft.irfft(x_,n=d,dim=2)

    elif choose=="Low3":
        x_= x_fft_init
        for i in range(0, int(fc1 * 0.85)):
            x_[:,:,i] = 0
            zeroo.append(i)
        x2=torch.fft.irfft(x_,n=d,dim=2)

    elif choose=="Low4":
        x_= x_fft_init
        for i in range(0, int(fc1 * 0.8)):
            x_[:,:,i] = 0
            zeroo.append(i)
        x2=torch.fft.irfft(x_,n=d,dim=2)

    elif choose=="Low5":
        x_= x_fft_init
        for i in range(0, int(fc1 * 0.75)):
            x_[:,:,i] = 0
            zeroo.append(i)
        x2=torch.fft.irfft(x_,n=d,dim=2)


    return x2,zeroo    
 