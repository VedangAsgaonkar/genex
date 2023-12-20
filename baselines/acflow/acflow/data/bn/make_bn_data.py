'''
make synthetic data from Bayesian Networks [1].

[1] https://www.cse.huji.ac.il/~galel/Repository/
'''

import numpy as np
import pickle
from pgmpy.readwrite import BIFReader

np.random.seed(1234)

def get_graph(fname, tar_list=None):
    reader = BIFReader(fname)
    nodes = reader.get_variables()
    candidates = set(nodes)
    print(nodes)
    edges = reader.get_edges()
    n_nodes = len(nodes)
    node2id = {node:id for id, node in enumerate(nodes)}
    for i, tar in enumerate(tar_list): 
        # select one as the target
        target = tar or np.random.choice(list(candidates))
        print('target: ', target)
        candidates -= {target}
        node2id[target], node2id[nodes[-(i+1)]] = node2id[nodes[-(i+1)]], node2id[target]
    print(node2id)
    # build adjmat
    graph = np.zeros([n_nodes, n_nodes])
    for edge in edges:
        p = node2id[edge[0]]
        n = node2id[edge[1]]
        graph[n, p] = 1
    print(graph)

    return graph

def gen_data(fname, tar_list=None, data_points=20000, sigma=1.0):
    graph = get_graph(fname, tar_list)
    n_features = n_nodes = len(graph)
    missing = {i:n for i, n in enumerate(graph.copy().sum(axis=1))}
    weights = np.random.uniform(0.1, 1.0, [n_features, n_features])
    weights *= graph
    data = np.zeros([data_points, n_features])
    while missing:
        candidates = [k for k in missing if missing[k]==0]
        f = data.copy() # [N,f]
        w = weights[candidates].copy() # [a,f]
        sam = np.matmul(f, w.T)
        data[:,candidates] = sam + np.random.randn(*sam.shape) * sigma
        for c in candidates:
            missing.pop(c)
        for k in missing:
            missing[k] -= sum(graph[k,candidates])

    return graph, weights, data

def main(fname, sname)
    tar_list = [None]
    data_points = 20000
    sigma = 0.3
    graph, weights, data = gen_data(fname, tar_list, data_points, sigma)
    graph = graph.astype(np.float32)
    data = data.astype(np.float32)
    x = data[:,:-1]
    y = data[:,-1:]

    n_train = int(data_points * 0.8)
    n_valid = n_test = int(data_points * 0.1)

    indices = list(range(data_points))
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:-n_test]
    test_indices = indices[-n_test:]

    train_x, train_y = x[train_indices], y[train_indices]
    valid_x, valid_y = x[valid_indices], y[valid_indices]
    test_x, test_y = x[test_indices], y[test_indices]

    data_dict = {
        'graph': graph,
        'weights': weights,
        'train': (train_x, train_y),
        'valid': (valid_x, valid_y),
        'test': (test_x, test_y)
    }

    with open(sname, 'wb') as f:
        pickle.dump(data_dict, f)

if __name__ == '__main__':
    fname = 'asia.bif'
    sname = 'asia_bn_0.3.pkl'
    main(fname, sname)

    fname = 'sachs.bif'
    sname = 'sachs_bn_0.3.pkl'
    main(fname, sname)