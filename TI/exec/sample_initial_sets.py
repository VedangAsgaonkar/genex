import argparse
import pickle
import numpy as np
import sys
sys.path.insert(1, "./data")
from costs import get_costs
import random

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='disease_pred', choices=['disease_pred','mnist','cifar100','tiny_imagenet'])
parser.add_argument('-ns', '--num_sets', type=int, default=100, help='Number of initial sets to consider')
parser.add_argument('-ne', '--num_elements', type=int, default=10, help='Number of elements in initials sets')
parser.add_argument('-sm', '--sampling_method', type=str, default='uniform', choices=['uniform', 'cost_prob', 'fixed_length'], help='sampling method')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

def sample_set_uniform(costs, initial_threshold):
    total_cost=0
    num_features = len(costs)
    initial_set = []
    max_c, min_c = initial_threshold
    while True:
        idx = np.random.choice(np.arange(num_features), replace=False)
        c = costs[idx]
        if (total_cost + c) < min_c:
            total_cost+=c
            initial_set.append(idx)
        elif (total_cost + c) >= min_c and (total_cost + c) <= max_c : 
            total_cost+=c
            initial_set.append(idx)
            return initial_set
        else:
            continue

def sample_sets_fixed_length(costs, initial_threshold):

    num_features = len(costs)
    print("Num_features = ", num_features)
    features = list(range(num_features))
    initial_set = random.sample(features, k = args.num_elements)
    return initial_set

def sample_set_cost_prob(costs, initial_threshold):
    total_cost=0
    num_features = len(costs)
    initial_set = []
    min_c, max_c = initial_threshold
    probs = (1 - costs/np.sum(costs))
    while True:
        idx = np.random.choice(np.arange(num_features), p=probs, replace=False)
        c = costs[idx]
        if (total_cost + c) < min_c:
            total_cost+=c
            initial_set.append(idx)
        elif (total_cost + c) >= min_c and (total_cost + c) <= max_c : 
            total_cost+=c
            initial_set.append(idx)
            return initial_set
        else:
            continue

# costs, initial_threshold = np.array(get_costs(args.data))
costs = get_costs(args.data)
costs, initial_threshold = np.array(costs[0]), (costs[1], costs[2])


initial_sets = []   # an array of features indices

if args.sampling_method=='uniform':
    sample_set = sample_set_uniform
elif args.sampling_method=='cost_prob':
    sample_set = sample_set_cost_prob
elif args.sampling_method=='fixed_length':
    sample_set = sample_sets_fixed_length
else:
    print('sampling method does not exist')
    exit(1)

np.random.seed(args.seed)
for i in range(args.num_sets):
    initial_sets.append(sample_set(costs, initial_threshold))


with open(f'data/initial_sets_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'wb') as f:
    pickle.dump(initial_sets, f)