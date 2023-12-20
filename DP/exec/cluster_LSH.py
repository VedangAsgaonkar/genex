import argparse
import pickle
import time
import numpy as np
import sys
import torch
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from LSH import LSH

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

usage_factor = 10

if args.data=='disease_pred':
    train_dataset = datasets.DiseasePredDataset("data/disease_pred/Training.csv")
    val_dataset = datasets.DiseasePredDataset("data/disease_pred/Testing.csv")
 
with open(f'data/initial_sets_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    initial_sets = pickle.load(f)

all_masked_samples = []
all_samples = []
all_labels = []
all_sets = []
non_zero_samples = []
all_indices = []

for s in initial_sets:
    masked_samples, samples, labels = get_samples_with_features(s, train_dataset)
    non_zero_indices = torch.where(masked_samples.sum(axis=1) != 0)[0]
    all_masked_samples += masked_samples[non_zero_indices]
    all_samples+=samples[non_zero_indices]
    all_labels+=labels[non_zero_indices]
    all_sets+=([s]*len(non_zero_indices))
    all_indices+= list(np.arange(len(train_dataset))) 

print(len(all_samples), len(train_dataset), len(initial_sets))

zeros = 0
for sample in all_samples:
    if(sample.sum() == 0):
        zeros += 1
    else:
        non_zero_samples.append(sample)

print("Total:", len(all_samples))
print("Zeros: ", zeros)


cluster_hyps = {
    "disease_pred" : {"hash_code_dim" : 32, "subset_size" : 4, "embed_dim" : 132, "num_hash_tables" : 1}
}

print('clustering...')
s=time.time()
all_samples = torch.stack(all_samples)
all_masked_samples = torch.stack(all_masked_samples)
non_zero_samples = torch.stack(non_zero_samples)
lsh = LSH(**cluster_hyps['disease_pred'])
lsh.index_corpus(all_masked_samples)
print(f'Done! Time taken = {time.time()-s}')


hash_tables = lsh.all_hash_tables[0]
cluster_labels = torch.zeros(all_samples.shape[0])
for key in hash_tables:
    for i in hash_tables[key]:
        cluster_labels[i] = key

pkl_data = {}
pkl_data["all_samples"] = all_samples
pkl_data["all_labels"] = torch.stack(all_labels)
pkl_data["all_sets"] = (all_sets)
pkl_data["all_indices"] = (all_indices)
pkl_data["cluster_labels"] = cluster_labels



with open(f'data/clusters_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'wb') as f:
    pickle.dump(pkl_data, f)

with open(f'data/clusterobj_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'wb+') as f:
    pickle.dump(lsh, f)

