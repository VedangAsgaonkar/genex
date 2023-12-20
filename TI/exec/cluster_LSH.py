'''
Script to cluster all the training samples based on all the initial sets
* exec/sample_initial_sets.py must be executed before running this script

run from src using command:
python exec/cluster.py
'''

import argparse
import pickle, os
import time
import numpy as np
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from LSH import LSH
import math
import matplotlib.pyplot as plt
import copy

sys.path.insert(1, "./data")
import datasets

sys.path.insert(1, "./")
from utils import get_samples_with_features

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, default='disease_pred', choices=['disease_pred','mnist','cifar100','tiny_imagenet'])
parser.add_argument('-ns', '--num_sets', type=int, default=20, help='Number of initial sets to consider')
parser.add_argument('-sm', '--sampling_method', type=str, default='fixed_length', choices=['uniform', 'cost_prob', 'fixed_length'], help='sampling method')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


if args.data=='cifar100':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
    ])
    train_dataset = torchvision.datasets.CIFAR100('data/cifar100', download=True, train=True, transform=transform_train)
    val_dataset = torchvision.datasets.CIFAR100('data/cifar100', download=True, train=False, transform=transform_test)


if args.data=='tiny_imagenet':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) # meanstd transformation
    dataset = datasets.TinyImageNet(root = "data/tiny_imagenet/", download = True, transform = transform_train)
    # train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [int(0.7*len(dataset)), int(0.3*len(dataset)), int(0.9*len(dataset))])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.7*len(dataset)), int(0.3*len(dataset))])
    print("train set length:", len(train_dataset))
    print("val set length:", len(val_dataset))
    # train_dataset = torchvision.datasets.CIFAR100('data/cifar100', download=True, train=True, transform=transform_train)
    # val_dataset = torchvision.datasets.CIFAR100('data/cifar100', download=True, train=False, transform=transform_test)


with open(f'data/initial_sets_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    initial_sets = pickle.load(f)

bs = len(train_dataset)
train_dataset = torch.utils.data.ConcatDataset([train_dataset, copy.deepcopy(train_dataset), copy.deepcopy(train_dataset), copy.deepcopy(train_dataset), copy.deepcopy(train_dataset)])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=False)

def getT2(T):
  out = []
  for i in T:
    if(args.data == "tiny_imagenet"):
        base = 256*(i//16) + 4*(i%16)
        for i in range(4):
            out += [base, base+1, base + 2, base + 3] 
            base += 64
    else:
        base = 64*(i//16)+2*(i%16)
        out += [base, base+1,base+32,base+33]
  return out

all_masked_samples = []
all_sets = []


# for tinyimagenet, length of s is 32*32
for batch_idx, (inputs, targets) in enumerate(trainloader):
    s = initial_sets[batch_idx]
    print("batchidx = ", batch_idx, "inputs shape:'", inputs.shape)
    print(targets.shape)
    mask = torch.zeros((inputs.shape[0],inputs.shape[1], inputs.shape[2]*inputs.shape[3]))
    mask[:,:,getT2(s)] = 1
    mask = mask.reshape(inputs.shape)
    masked_samples = inputs*mask
    all_masked_samples.append(masked_samples.reshape(-1,12288))
    all_sets+=([s]*inputs.shape[0])

all_masked_samples = torch.cat(all_masked_samples)

print(all_masked_samples.shape, len(train_dataset), len(initial_sets))

## Need to carefully decide these hyperparams
cluster_hyps = {
    "disease_pred" : {"hash_code_dim" : 32, "subset_size" : 4, "embed_dim" : 132, "num_hash_tables" : 1},
    "mnist" : {"hash_code_dim" : 32, "subset_size" : 3, "embed_dim" : 784, "num_hash_tables" : 1},
    "cifar100" : {"hash_code_dim" : 32, "subset_size" : 2, "embed_dim" : 3072, "num_hash_tables" : 1},
    "tiny_imagenet" : {"hash_code_dim" : 32, "subset_size" : 2, "embed_dim" : 12288, "num_hash_tables" : 1}
}

print('clustering...')
s=time.time()
# clustering = DBSCAN(min_samples=cluster_hyps[args.data]["min_samples"], eps=cluster_hyps[args.data]["eps"], metric='cosine', n_jobs=4).fit(all_samples)
# print(AgglomerativeClustering.__defaults_)
lsh = LSH(**cluster_hyps[args.data])
lsh.index_corpus(all_masked_samples)
# clustering = AgglomerativeClustering(n_clusters=cluster_hyps[args.data]["n_clusters"], metric="cosine", linkage="average").fit(np.array(all_masked_samples))
# clustering = KMeans(n_clusters=cluster_hyps[args.data]["n_clusters"]).fit(np.array(non_zero_samples))
print(f'Done! Time taken = {time.time()-s}')


# cluster_labels = clustering.labels_
hash_tables = lsh.all_hash_tables[0]
cluster_labels = torch.zeros(all_masked_samples.shape[0])
indices = [[],[],[],[]]
for key in hash_tables:
    for i in hash_tables[key]:
        cluster_labels[i] = key
        indices[int(key)].append(i)


pkl_data = {}
pkl_data["train_dataset"] = train_dataset
pkl_data["train_sets"] = (all_sets)
pkl_data["train_cluster_indices"] = indices

bs = len(val_dataset)
val_dataset = torch.utils.data.ConcatDataset([val_dataset, copy.deepcopy(val_dataset), copy.deepcopy(val_dataset), copy.deepcopy(val_dataset), copy.deepcopy(val_dataset)])
testloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

val_all_masked_samples = []
val_all_sets = []

for batch_idx, (inputs, targets) in enumerate(testloader):
    s = initial_sets[batch_idx]
    mask = torch.zeros((inputs.shape[0],inputs.shape[1], inputs.shape[2]*inputs.shape[3]))
    mask[:,:,getT2(s)] = 1
    mask = mask.reshape(inputs.shape)
    masked_samples = inputs*mask
    val_all_masked_samples.append(masked_samples.reshape(-1,12288))
    val_all_sets+=([s]*inputs.shape[0])

val_all_masked_samples = torch.cat(val_all_masked_samples)
val_cluster_labels = []
print(val_all_masked_samples.shape, len(val_dataset), len(initial_sets))
for s in val_all_masked_samples:
    val_cluster_labels.append(lsh.retrieve(s))

val_indices = [[],[],[],[]]
for i in range(len(val_cluster_labels)):
    val_indices[int(val_cluster_labels[i])].append(i)

pkl_data["test_dataset"] = val_dataset
pkl_data["test_sets"] = val_all_sets
pkl_data["test_cluster_indices"] = val_indices

with open(f'data/clusters_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'wb') as f:
    pickle.dump(pkl_data, f)

with open(f'data/clusterobj_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'wb+') as f:
    pickle.dump(lsh, f)