from torch.utils.data import Dataset    
import numpy as np
import torch
import torch.utils.data
import torch
import random 
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MyDataset(Dataset):
    def __init__(self, x_data, label,data_index):
        super(MyDataset, self).__init__()
        x_data = x_data.astype(np.float32)
        x_data[np.isnan(x_data)]  = 0
        self.x_data =torch.Tensor(x_data)
        self.y_data = torch.Tensor(label)
        self.id=torch.IntTensor(data_index)
    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        time0_tuple = index
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                time1_tuple = random.randint(0,len(self.id)-1)
                if self.y_data[time0_tuple]==self.y_data[time1_tuple]:
                    break
        else:
            while True:
                #keep looping till a different class image is found         
                time1_tuple = random.randint(0,len(self.id)-1)
                if self.y_data[time0_tuple]!=self.y_data[time1_tuple]:
                    break

        return self.x_data[time0_tuple].to(device), self.x_data[time1_tuple].to(device) , torch.from_numpy(np.array([int(self.y_data[time0_tuple]!=self.y_data[time1_tuple])],dtype=np.float32)).to(device),self.y_data[time0_tuple].to(device), self.y_data[time1_tuple].to(device)
