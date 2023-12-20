'''
preprocess UCI datasets
please first download original datasets from [1]

[1] https://archive.ics.uci.edu/ml/datasets.php
'''

import numpy as np
import pickle


# boston housing
data = np.loadtxt('/path/to/original/data', dtype=np.float32)
data += np.random.randn(*data.shape) * 1e-2
dmin = data.min(axis=0)
dmax = data.max(axis=0)
data = (data - dmin) / (dmax - dmin)

ind = np.arange(len(data))
np.random.shuffle(ind)
n_test = n_valid = int(len(data) * 0.1)
n_train = len(data) - n_valid - n_test

train_data = data[ind[:n_train]]
valid_data = data[ind[n_train:-n_test]]
test_data = data[ind[-n_test:]]

train_x, train_y = train_data[:,:-1], train_data[:,-1:]
valid_x, valid_y = valid_data[:,:-1], valid_data[:,-1:]
test_x, test_y = test_data[:,:-1], test_data[:,-1:]

data = {
    'train': (train_x, train_y),
    'valid': (valid_x, valid_y),
    'test': (test_x, test_y)
}

with open('housing.pkl', 'wb') as f:
    pickle.dump(data, f)


# white wine
data = np.loadtxt('/path/to/original/data', skiprows=1, delimiter=';')
data = data.astype(np.float32)
dmin = data.min(axis=0)
dmax = data.max(axis=0)
data = (data - dmin) / (dmax - dmin)

ind = np.arange(len(data))
np.random.shuffle(ind)
n_test = n_valid = int(len(data) * 0.1)
n_train = len(data) - n_valid - n_test

train_data = data[ind[:n_train]]
valid_data = data[ind[n_train:-n_test]]
test_data = data[ind[-n_test:]]

train_x, train_y = train_data[:,:-1], train_data[:,-1:]
valid_x, valid_y = valid_data[:,:-1], valid_data[:,-1:]
test_x, test_y = test_data[:,:-1], test_data[:,-1:]

data = {
    'train': (train_x, train_y),
    'valid': (valid_x, valid_y),
    'test': (test_x, test_y)
}

with open('./whitewine.pkl', 'wb') as f:
    pickle.dump(data, f)
