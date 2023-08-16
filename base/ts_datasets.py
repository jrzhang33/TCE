
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Dataset
from sktime.datasets import load_UCR_UEA_dataset
from sklearn.utils import shuffle
from .ts_utils import load_UCR_UEA_dataset_from_tsfile, noise_mask


class TS_TAGDataset(Dataset):
    "Given a specific tsDataset and tag, take out the dataset"
    "dataset: data, label"

    def __init__(self, baseset, label):
        labels = np.unique([baseset[i][1] for i in range(len(baseset))])
        assert label in labels
        super().__init__()
        self.tagset = [baseset[i][0] for i in range(len(baseset)) if baseset[i][1] == label]

    def __len__(self):
        return len(self.tagset)
    
    def __getitem__(self, index):
        return self.tagset[index]


class TS_GENDataset(Dataset):
    "Given a specific tsDataset and tag, take out the dataset"
    "dataset: data, label"

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, index):
        return self.data[index], torch.tensor(self.label)


def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data


def dataframe2ndarray(X):

    all_len = [len(x) for x in X]
    max_len = max(all_len)

    _X = []
    for x in X:

        if x.isnull().any():
            x = x.interpolate(method='linear', limit_direction='both')

        if len(x) < max_len:
            x = uniform_scaling(x, max_len)
        _X.append(x)
    _X = np.array(np.transpose(_X))

    return _X

def get_max_seq_len(data_df):

    max_seq_len = 0
    for i in range(len(data_df)):
        X = data_df.iloc[i, :].copy(deep=True)
        max_seq_len = max(max_seq_len, max([len(x) for x in X]))
    return max_seq_len


class tsNormlizer:

    def __init__(self, mode='minmax', scale=(0., 1.)):
        assert mode in ['minmax', 'standard', None]

        self.mode = mode
        self.scale = scale

    def fit(self, X):
        self.data_max_ = []
        self.data_min_ = []
        self.data_mean_ = []
        self.data_std_ = []
        for dim in X.columns:
            x = X[dim]
            total_x = []
            for _x in x:
                total_x.extend(list(_x))
            self.data_max_.append(max(total_x))
            self.data_min_.append(min(total_x))
            self.data_mean_.append(np.mean(total_x))
            self.data_std_.append(np.std(total_x))

    def transform(self, x):
        result = []
        for i in range(x.shape[-1]):
            _x = x[:, i]
            if self.mode == 'minmax':
                _x = (_x - self.data_min_[i]) / (self.data_max_[i] - self.data_min_[i])
                _x = self.scale[0] + _x * (self.scale[1] - self.scale[0])
            elif self.mode == 'standard':
                _x = (_x - self.data_mean_[i]) / self.data_std_[i]
            
            result.append(_x[:, np.newaxis])
        
        return np.concatenate(result, axis=-1)


class UCR_UEADataset(Dataset):
    "Torch Datasets for UCR/UEA archive"

    def __init__(self, name, split=None, extract_path="ucr_uea_archive", return_y=True, padding='zero', normalize=None, channel_first=False):
        assert split in ['train', 'test', None]
        assert normalize in ['standard', 'minmax', None]
        assert padding in ['zero', 'mean', None]

        super().__init__()
        self.return_y = return_y
        self.padding = padding
        self.normalize = normalize
        self.channel_first = channel_first


        self.data, self.label = load_UCR_UEA_dataset_from_tsfile(name, extract_path=extract_path, split=split)
        data_all, _ = load_UCR_UEA_dataset_from_tsfile(name, extract_path=extract_path, split=None)
        self.max_len = get_max_seq_len(data_all) 
        self.normalizer = tsNormlizer(normalize, scale=(0, 1))
        self.normalizer.fit(self.data)


        self.label2y = dict([(y, i) for i, y in enumerate(np.unique(self.label))])
        self.y2label = list(self.label2y.values()) 
        self.y = [self.label2y[label] for label in self.label] 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data.iloc[idx].copy(deep=True)
        X = dataframe2ndarray(X)    

        if self.normalize is not None:
            X = self.normalizer.transform(X)

        if self.padding == 'zero':
            pad = np.zeros((self.max_len - X.shape[0], X.shape[1])) # [PAD]
            X = np.concatenate([X, pad], axis=0)
        elif self.padding == 'mean':
            pad = []
            for i in range(X.size(1)):
                _p = np.ones((self.max_len - X.shape[0], 1)) * self.normalizer.data_mean_[i]
                pad.append(_p)
            pad = np.concatenate(pad, dim=1)
            X = np.concatenate([X, pad], axis=0)

        item = {'input': torch.from_numpy(X).float()}
        if self.return_y:
            item['label'] = torch.tensor(self.y[idx]).long()
        
        if self.channel_first:
            item['input'] = item['input'].transpose(-1, -2)
        
        return item


class Transformer_UCR_UEADataset(Dataset):
    "Torch Datasets for UCR/UEA archive"

    def __init__(self, name, split=None, add_cls=True, extract_path="ucr_uea_archive", max_len=256, return_y=True, mask=False, 
        normalize=None):
        assert split in ["train", "test", None]
        assert normalize in ["standard", "minmax", None]

        super().__init__()
        self.return_y = return_y
        self.mask = mask
        self.normalize = normalize
        self.add_cls = add_cls

        self.data, self.label = load_UCR_UEA_dataset(name, split=split, return_X_y=True, \
            extract_path=extract_path) # x, y => Dataframe
        
        self.max_len = max(max_len, get_max_seq_len(self.data) + 1) 
        self.normalizer = tsNormlizer(scale=(0.05, 0.95))
        self.normalizer.fit(self.data)

        self.label = np.array(self.label) 
        self.label2y = dict([(y, i) for i, y in enumerate(np.unique(self.label))])
        self.y2label = list(self.label2y.values()) 
        self.y = [self.label2y[label] for label in self.label] 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data.iloc[idx].copy(deep=True)
        X = dataframe2ndarray(X)  

        X = self.normalizer.transform(X)
        if self.add_cls:
            padding_mask = [0] + [0] * X.shape[0] + [1] * (self.max_len - X.shape[0] - 1)
            cls = np.ones((1, X.shape[-1])) # [CLS]
            pad = np.zeros((self.max_len - X.shape[0] - 1, X.shape[-1])) # [PAD]
            X = np.concatenate([cls, X, pad], axis=0)
        else:
            padding_mask = [0] * X.shape[0] + [1] * (self.max_len - X.shape[0])
            pad = np.zeros((self.max_len - X.shape[0], X.shape[-1])) # [PAD]
            X = np.concatenate([X, pad], axis=0)

        item = {"input": torch.from_numpy(X).float(), "padding_mask": torch.tensor(padding_mask).bool()}

        if self.return_y:
            item["label"] = torch.tensor(self.y[idx]).long()
        
        return item


class MLM_UCR_UEADataset(Dataset):
    "Torch Datasets for UCR/UEA archive"

    def __init__(self, name, split=None, pt_ratio=0.5, extract_path="ucr_uea_archive", \
        max_len=256, normalize=None, masking_ratio=0.2, lm=5, mode='separate', distribution='geometric'):

        assert split in ["train", "test", None]
        assert normalize in ["standard", "minmax", None]

        super().__init__()
        self.pt_ratio = pt_ratio
        self.normalize = normalize
        self.masking_ratio = masking_ratio
        self.lm = lm
        self.mode = mode
        self.distribution = distribution

        self.data, _ = load_UCR_UEA_dataset(name, split=split, return_X_y=True, \
            extract_path=extract_path) # x, y => Dataframe
        
        self.data = shuffle(self.data).reset_index(drop=True)
        self.data = self.data.iloc[: int(len(self.data)*self.pt_ratio)]

        self.max_len = max(max_len, get_max_seq_len(self.data) + 1) 
        self.normalizer = tsNormlizer(scale=(0.05, 0.95))
        self.normalizer.fit(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data.iloc[idx].copy(deep=True)
        # print(X)
        X = dataframe2ndarray(X)    

        X = self.normalizer.transform(X)
        # padding mask
        padding_mask = [0] + [0] * X.shape[0] + [1] * (self.max_len - X.shape[0] - 1)
        # lm mask
        lm_mask = ~noise_mask(X, self.masking_ratio, self.lm, self.mode, self.distribution)

        cls = np.ones((1, X.shape[-1])) # [CLS]
        pad = np.zeros((self.max_len - X.shape[0] - 1, X.shape[-1])) # [PAD]
        X = np.concatenate([cls, X, pad], axis=0)
        
        # lm_mask
        cls_mask = np.zeros((1, X.shape[-1]), dtype=np.bool) # [CLS]
        pad_mask = pad[:]
        lm_mask = torch.from_numpy(np.concatenate([cls_mask, lm_mask, pad_mask], axis=0)).bool()

        item = {"input": torch.from_numpy(X[:]).masked_fill(lm_mask, -1).float(), \
            "padding_mask": torch.tensor(padding_mask).bool(), 
            "output": torch.from_numpy(X[:]).float(), 
            "lm_mask": lm_mask}
        
        return item


class tsMinMaxNormlizer:

    def __init__(self, scale=(0, 1)):
        self.scale = scale

    def fit(self, X):

        self.data_max_ = np.max(X.reshape(-1, X.shape[-1]), axis=0)
        self.data_min_ = np.min(X.reshape(-1, X.shape[-1]), axis=0)

    def transform(self, x):

        result = []
        for i in range(x.shape[-1]):
            _x = x[:, i]
            _x = (_x - self.data_min_[i]) / (self.data_max_[i] - self.data_min_[i])
            _x = self.scale[0] + _x * (self.scale[1] - self.scale[0])
            result.append(_x[:, np.newaxis])
        
        return np.concatenate(result, axis=-1)


class Soil_Dataset(Dataset):
    "Torch Datasets for UCR/UEA archive"

    def __init__(self, data, label, return_y=True, normalize='minmax', max_len=64, channel_first=False):
        assert normalize in ['standard', 'minmax', None]

        super().__init__()
        self.normalize = normalize
        self.data, self.y = data, label
        self.return_y = return_y
        self.max_len = max_len
        self.channel_first = channel_first
        self.normalizer = tsMinMaxNormlizer(scale=(0.0, 1.0))
        self.normalizer.fit(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx]

        if self.normalize is not None:
            X = self.normalizer.transform(X)

        item = {'input': torch.from_numpy(X).float()}
        if self.return_y:
            item['label'] = torch.tensor(self.y[idx]).long()
        
        if self.channel_first:
            item['input'] = item['input'].transpose(-1, -2)
        
        return item


class MLM_SoilDataset(Dataset):
    "Torch Datasets for UCR/UEA archive"

    def __init__(self, data, label, normalize='minmax', max_len=64, \
        masking_ratio=0.2, lm=5, mode='separate', distribution='geometric'):

        assert normalize in ["standard", "minmax", None]

        super().__init__()
        # self.pt_ratio = pt_ratio
        self.normalize = normalize
        self.data, self.y = data, label
        self.masking_ratio = masking_ratio
        self.lm = lm
        self.mode = mode
        self.distribution = distribution
        self.normalizer = tsMinMaxNormlizer(scale=(0.0, 1.0))
        self.normalizer.fit(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx]

        if self.normalize is not None:
            X = self.normalizer.transform(X)

        item = {'input': torch.from_numpy(X).float()}
        
        # padding mask
        # padding_mask = [0] * X.shape[0] + [1] * (self.max_len - X.shape[0] - 1)
        # lm mask
        lm_mask = ~noise_mask(X, self.masking_ratio, self.lm, self.mode, self.distribution)
        lm_mask = torch.tensor(lm_mask).long()

        item = {"input": torch.from_numpy(X[:]).masked_fill(lm_mask, -1).float(), \
            "output": torch.from_numpy(X[:]).float(), 
            "lm_mask": lm_mask}
        
        return item
