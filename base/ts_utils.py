import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sktime.datasets._data_io import load_from_tsfile_to_dataframe

def ucrdf_to_tensor(df):

    ans = []
    for c in df.columns:
        ans.append(torch.Tensor(df[c]).unsqueeze(1))
    
    return torch.cat(ans, dim=1)

def ucrdf_to_nparray(df):
    ans = []
    for c in df.columns:
        ans.append(np.expand_dims(np.array(df[c].tolist()), axis=1))
    
    return np.concatenate(ans, axis=1)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

def noise_mask(X, masking_ratio, lm=3, mode='separate', distribution='geometric', exclude_feats=None):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    if exclude_feats is not None:
        exclude_feats = set(exclude_feats)

    if distribution == 'geometric':  # stateful (Markov chain)
        if mode == 'separate':  # each variable (feature) is independent
            mask = np.ones(X.shape, dtype=bool)
            for m in range(X.shape[1]):  # feature dimension
                if exclude_feats is None or m not in exclude_feats:
                    mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
        else:  # replicate across feature dimension (mask all variables at the same positions concurrently)
            mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])
    else:  # each position is independent Bernoulli with p = 1 - masking_ratio
        if mode == 'separate':
            mask = np.random.choice(np.array([True, False]), size=X.shape, replace=True,
                                    p=(1 - masking_ratio, masking_ratio))
        else:
            mask = np.tile(np.random.choice(np.array([True, False]), size=(X.shape[0], 1), replace=True,
                                            p=(1 - masking_ratio, masking_ratio)), X.shape[1])

    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked

    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask

def load_UCR_UEA_dataset_from_tsfile(name, extract_path, split=None):
    assert split in ['train', 'test', None]
    if extract_path == 'Multi':
        if split in ['train', 'test']:
            split = split.upper()
            file_path = f'./datasets/Multivariate2018_ts/Multivariate_ts/{name}//{name}_{split}.ts'
            X, y = load_from_tsfile_to_dataframe(file_path)
        else:
            X_train, y_train = load_from_tsfile_to_dataframe(f'./datasets/Multivariate2018_ts/Multivariate_ts/{name}//{name}_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe(f'./datasets/Multivariate2018_ts/Multivariate_ts/{name}//{name}_TEST.ts')
            X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            y = np.concatenate([y_train, y_test], axis=0)
    else:
        if split in ['train', 'test']:
            split = split.upper()
            file_path = f'./datasets/Univariate2018_ts/Univariate_ts/{name}//{name}_{split}.ts'
            X, y = load_from_tsfile_to_dataframe(file_path)
        else:
            X_train, y_train = load_from_tsfile_to_dataframe(f'./datasets/Univariate2018_ts/Univariate_ts/{name}//{name}_TRAIN.ts')
            X_test, y_test = load_from_tsfile_to_dataframe(f'./datasets/Univariate2018_ts/Univariate_ts/{name}//{name}_TEST.ts')
            X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            y = np.concatenate([y_train, y_test], axis=0)        
        
    return X, y
