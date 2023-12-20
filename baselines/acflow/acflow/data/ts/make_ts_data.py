'''
preprocesse time series data.
please download the original data from [1]

[1] http://www.timeseriesclassification.com/index.php
'''

import numpy as np
import pickle
import pandas as pd

from reader import load_from_tsfile_to_dataframe

# pedestrian dataset
dir_name = '/path/to/original/data'
train_file = dir_name + 'MelbournePedestrian_TRAIN.ts'
test_file = dir_name + 'MelbournePedestrian_TEST.ts'

train_x, train_y = load_from_tsfile_to_dataframe(train_file)
test_x, test_y = load_from_tsfile_to_dataframe(test_file)

train_x_np = []
train_y_np = train_y.astype(np.int32)
for i in range(len(train_x)):
    x_nda = train_x.iloc[i]['dim_0'].to_numpy() 
    train_x_np.append(x_nda)
train_x_np = np.stack(train_x_np)
inds = np.where(np.isnan(train_x_np))[0]
train_x_np = np.delete(train_x_np, inds, axis=0)
train_y_np = np.delete(train_y_np, inds, axis=0)
assert not np.any(np.isnan(train_x_np))
print(train_x_np.shape, train_y_np.shape)
print(np.unique(train_y_np))

test_x_np = []
test_y_np = test_y.astype(np.int32)
for i in range(len(test_x)):
    x_nda = test_x.iloc[i]['dim_0'].to_numpy()
    if x_nda.shape[0] != 24:
        test_y_np = np.delete(test_y_np, i, axis=0)
        continue
    test_x_np.append(x_nda)
test_x_np = np.stack(test_x_np)
inds = np.where(np.isnan(test_x_np))[0]
test_x_np = np.delete(test_x_np, inds, axis=0)
test_y_np = np.delete(test_y_np, inds, axis=0)
assert not np.any(np.isnan(test_x_np))
print(test_x_np.shape, test_y_np.shape)
print(np.unique(test_y_np))

x_mean = train_x_np.mean(axis=0)
x_std = train_x_np.std(axis=0)

train_x = (train_x_np - x_mean) / x_std
train_x = np.expand_dims(train_x, axis=-1)
train_y = train_y_np - 1

test_x = (test_x_np - x_mean) / x_std
test_x = np.expand_dims(test_x, axis=-1)
test_y = test_y_np - 1

data_dict = {
    'train': (train_x, train_y),
    'valid': (test_x, test_y),
    'test': (test_x, test_y)
}
with open('./predestrian.pkl', 'wb') as f:
    pickle.dump(data_dict, f)


# digits
dir_name = '/path/to/original/data'
train_file = dir_name + 'PenDigits_TRAIN.ts'
test_file = dir_name + 'PenDigits_TEST.ts'

train_x, train_y = load_from_tsfile_to_dataframe(train_file)
test_x, test_y = load_from_tsfile_to_dataframe(test_file)

train_x_np = []
train_y_np = train_y.astype(np.int32)
for i in range(len(train_x)):
    x0_nda = train_x.iloc[i]['dim_0'].to_numpy()
    x1_nda = train_x.iloc[i]['dim_1'].to_numpy()
    x_nda = np.stack([x0_nda, x1_nda], axis=1)
    train_x_np.append(x_nda)
train_x_np = np.stack(train_x_np)
assert not np.any(np.isnan(train_x_np))
print(train_x_np.shape, train_y_np.shape)
print(np.unique(train_y_np))

test_x_np = []
test_y_np = test_y.astype(np.int32)
for i in range(len(test_x)):
    x0_nda = test_x.iloc[i]['dim_0'].to_numpy()
    x1_nda = test_x.iloc[i]['dim_1'].to_numpy()
    x_nda = np.stack([x0_nda, x1_nda], axis=1)
    test_x_np.append(x_nda)
test_x_np = np.stack(test_x_np)
assert not np.any(np.isnan(test_x_np))
print(test_x_np.shape, test_y_np.shape)
print(np.unique(test_y_np))

train_x = ( train_x_np + np.random.rand(*train_x_np.shape) ) / 101.
test_x = ( test_x_np + np.random.rand(*test_x_np.shape) ) / 101.

train_y = train_y_np
test_y = test_y_np

print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

data_dict = {
    'train': (train_x, train_y),
    'valid': (test_x, test_y),
    'test': (test_x, test_y)
}
with open('./digits.pkl', 'wb') as f:
    pickle.dump(data_dict, f)
