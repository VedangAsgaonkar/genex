'''
make a synthetic dataset for classification.
'''

import numpy as np
import pickle

np.random.seed(12506)

def gen_data(n_features=10, data_points=10000, sigma=0.3):
    label = np.random.choice([-1.,1.], size=[data_points])
    data = np.random.randn(data_points, n_features) * sigma
    data[:,0] = np.random.rand(data_points)
    w1 = np.random.rand()
    w2 = np.random.rand()
    thresh = np.linspace(.1,1.,10)[:-1]
    for d,l in zip(data,label):
        mask = np.logical_and(d[0] >= thresh, d[0] < thresh+0.1)
        ind = np.where(mask)[0]
        d[ind+1] = w1*l + w2*d[0] + np.random.randn() * 0.01

    return data.astype(np.float32), label.astype(np.int64)

data, label = gen_data()
label = ((label + 1) // 2).astype(np.float32)
print(data.shape)
print(label.shape)
print(np.unique(label))
print(data[:5])
print(label[:5])

indices = list(range(10000))
np.random.shuffle(indices)
train_indices = indices[:8000]
valid_indices = indices[8000:9000]
test_indices = indices[9000:]

train_x, train_y = data[train_indices], label[train_indices]
valid_x, valid_y = data[valid_indices], label[valid_indices]
test_x, test_y = data[test_indices], label[test_indices]


data_dict = {
    'train': (train_x, train_y),
    'valid': (valid_x, valid_y),
    'test': (test_x, test_y)
}

with open('./syn.pkl', 'wb') as f:
    pickle.dump(data_dict, f)