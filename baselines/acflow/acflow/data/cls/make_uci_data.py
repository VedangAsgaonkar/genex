'''
preprocess UCI datasets.
please first download the original dataset from [1]

[1] https://archive.ics.uci.edu/ml/datasets.php
'''

import os
import numpy as np
from collections import Counter
import pickle
import pandas as pd

# gas
data = np.loadtxt('/path/to/original/data', dtype=np.float32, skiprows=1)
print(data.shape)
id2label = {}
label_dict = {
    'background': 0,
    'wine': 1,
    'banana': 2
}
label_file = '/path/to/label/file'
with open(label_file, 'r') as f:
    lines = f.readlines()
for line in lines[1:]:
    elems = line.split()
    id = float(elems[0])
    lab = label_dict[elems[2]]
    id2label[id] = lab
label = np.array([id2label[id] for id in data[:,0]])
print(label.shape)

data = data[:,2:].astype(np.float32)
label = label.astype(np.float32)

# normalize
minv = data.min(axis=0)
maxv = data.max(axis=0)
data = (data - minv) / (maxv - minv)

# split
N = len(data)
n_valid = n_test = int(0.1*N)
n_train = N-n_valid-n_test

indices = np.arange(N)
np.random.shuffle(indices)
train_ind = indices[:n_train]
valid_ind = indices[n_train:-n_test]
test_ind = indices[-n_test:]

train_x, train_y = data[train_ind], label[train_ind]
valid_x, valid_y = data[valid_ind], label[valid_ind]
test_x, test_y = data[test_ind], label[test_ind]

# save
data = {
    'train': (train_x, train_y),
    'valid': (valid_x, valid_y),
    'test': (test_x, test_y)
}
with open('gas.pkl', 'wb') as f:
    pickle.dump(data, f)

