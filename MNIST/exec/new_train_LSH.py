import argparse
import pickle
import numpy as np
import math
import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
torch.cuda.empty_cache()
  
sys.path.append(parent_directory)

from data import datasets
from data.costs import get_costs
from utils import *

from models.transformer import Transformer_Encoder, Transformer_Encoder_pytorch, Set_Embedding, Configuration, VAE_sample, ELBO_loss
from models.decoder import Decoder
from models.classifier import Classifier

parser = argparse.ArgumentParser()

# cluster related arguments
parser.add_argument('-d', '--data', type=str, default='disease_pred', choices=['disease_pred','mnist','cifar100','tiny_imagenet'])
parser.add_argument('-ns', '--num_sets', type=int, default=20, help='Number of initial sets to consider')
parser.add_argument('-sm', '--sampling_method', type=str, default='uniform', choices=['uniform', 'cost_prob', 'fixed_length'], help='sampling method')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-c','--cluster_id', type=int, default=0)

parser.add_argument('--train_gen', action='store_true', default=False)
# parser.add_argument('--set_size', type=int) # size of the universal set of features
parser.add_argument('-nc', '--num_classes', type=int, default=42) # number of classes
parser.add_argument('--p', type=float, default=0.5) # dropout value for generator training
parser.add_argument('--set_embedding_dim', type=int, default=69)
parser.add_argument('--encoder_output_dim', type=int, default=70)
parser.add_argument('--latent_dim', type=int, default=50) # latent dim for VAE
parser.add_argument('--num_heads', type=int, default=7)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--attention_dropout_rate',type=float,default=0.2)

# training related arguments    # set these carefully and similar to the baselines
parser.add_argument('--epochs', type=int, default=100)  # total epochs
parser.add_argument('-bs', '--batch_size', type=int, default=128)   
parser.add_argument('-lr', '--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('-g_freq', '--greedy_freq', type=int, default=2)
parser.add_argument('-s_freq', '--save_freq', type=int, default=5)
# parser.add_argument('-g_thres', '--greedy_thres', type=float, default=0.001, help='The percentage of loss improvement for a feature to be accepted') # was 0.05
parser.add_argument('--monte_carlo_trials', type=int, default=10)
parser.add_argument('--greedy_sample_size', type=int, default=3000)
parser.add_argument('--greedy_threshold', type=float, default=0.05)
parser.add_argument('--exploration_probability', type=float, default=0.1)

parser.add_argument('--budget', type=float, default=20)
parser.add_argument('--fix_ug', action='store_true')
parser.add_argument('--ug', type=float, default=0.4)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--train_vanilla',action='store_true', default=False)
parser.add_argument('--test_vanilla', action='store_true', default=False)
parser.add_argument('--train_x_S_only', action='store_true', default=False)
parser.add_argument('--train_random', action='store_true', default=False)
parser.add_argument('--train_fixed_T', action='store_true', default=False)
parser.add_argument('--train_breakout', action='store_true', default=False)
parser.add_argument('--end_to_end', action='store_true', default=False)
parser.add_argument('--warm_start', action='store_true', default=False)
parser.add_argument('--warm_end', action='store_true', default=False)
parser.add_argument('--warm_start_lr', type = float, default=0.001)
parser.add_argument('--warm_end_lr', type = float, default=0.001)
parser.add_argument('--test_on_train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--warm_end_end_to_end', action='store_true', default=False)
parser.add_argument('--warm_end_loss', type=str, default='cross_entropy', choices=['cross_entropy','generative_loss'])
parser.add_argument('--warm_end_fresh_start', action='store_true', default=False)
parser.add_argument('--no_greedy', action='store_true', default=False)
parser.add_argument('--warm_end_load_T', type=int, default=24)
parser.add_argument('--generate_during_test', action='store_true', default=False)


# MLP Arguments
parser.add_argument('--clf_hidden_sizes', help='clf mlp size',type=str, default='[32]')
parser.add_argument('--clf_p', help='dropout prob', type=float, default=0)
parser.add_argument('--batchnorm', action='store_true', help='batch norm')
parser.add_argument('--dropout', action='store_true', help='Dropout classifier')


parser.add_argument('-gpu', type=int, default=4)
parser.add_argument('-sf', '--save_folder', type=str, default='save_tensorized')
parser.add_argument('-smf', '--save_metrics_folder', type=str, default='save_metrics')

parser.add_argument('--generate_for_min', action='store_true', help='Do generation for min accuracy cluster')
# parser.add_argument('--remove_zeros', action='store_true', help='Remove zero samples from hard coded cluster id')

args = parser.parse_args()
args.clf_hidden_sizes = eval(args.clf_hidden_sizes)
print(args)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')
    
print('Using device:',device)

torch.random.manual_seed(0)
np.random.seed(0)

if args.data=='mnist':
    train_dataset = datasets.MnistDataset("data/mnist/Training.csv")
    val_dataset = datasets.MnistDataset("data/mnist/Testing.csv")
    args.norm = 256.0
elif args.data=='disease_pred':
    train_dataset = datasets.DiseasePredDataset("data/disease_pred/Training.csv")
    val_dataset = datasets.DiseasePredDataset("data/disease_pred/Testing.csv")
    args.norm = 1.0
else:
    raise NotImplementedError

args.set_size = train_dataset[0][0].shape[0]

## loading the clusters
with open(f'data/clusters_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    print(f'data/clusters_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl')
    pkl_data = pickle.load(f)
    all_samples = pkl_data["all_samples"]
    all_labels = pkl_data["all_labels"]
    all_sets = pkl_data["all_sets"]
    # all_indices = pkl_data["all_indices"]
    
    cluster_labels = pkl_data["cluster_labels"]

with open(f'data/clusterobj_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    val_cluster_obj = pickle.load(f)

with open(f'data/initial_sets_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    val_initial_sets = pickle.load(f)

if args.test:
    val_all_samples = []
    val_all_labels = []
    val_all_sets = []
    val_all_masked_samples = []

    for s in val_initial_sets:
        masked_samples, samples, labels = get_samples_with_features(s, val_dataset)
        non_zero_indices = torch.where(masked_samples.sum(axis = 1) != 0)[0]
        
        val_all_samples+=samples[non_zero_indices]
        val_all_labels+=labels[non_zero_indices]
        val_all_sets+=([s]*len(non_zero_indices))
        val_all_masked_samples+=masked_samples[non_zero_indices]

    val_all_samples = torch.stack(val_all_samples)
    val_all_masked_samples = torch.stack(val_all_masked_samples)
    val_cluster_labels = []
    for s in val_all_masked_samples:
        val_cluster_labels.append(val_cluster_obj.retrieve(s.cpu()))

    val_cluster_labels = np.array(val_cluster_labels)

config = Configuration(args.encoder_output_dim, args.num_layers, args.attention_dropout_rate, args.num_heads, args.latent_dim)

transformer_encoder = Transformer_Encoder_pytorch(args)
set_embedding = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
decoder = Decoder(args.latent_dim, args.set_size)
classifier = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
transformer_encoder = transformer_encoder.to(device)
set_embedding = set_embedding.to(device)
decoder = decoder.to(device)

all_samples = np.array(all_samples)
all_labels = np.array(all_labels)
print(all_samples.shape, all_labels.shape, cluster_labels.shape)
cluster_dict = {}
for l in np.unique(cluster_labels):
    sets = []
    ids = (cluster_labels==l)
    for i in np.arange(len(all_sets))[ids]:
        sets.append(all_sets[i])
    cluster_dict[l] = {
        "samples": all_samples[ids], 
        "labels": all_labels[ids],
        "sets": sets
        }

max_T = 0
s_indices = np.array((cluster_dict[args.cluster_id]["sets"]))
s_binary = np.zeros(all_samples.shape[1])


for i in range(s_indices.shape[0]):
    s_binary[s_indices[i]] += 1

s_binary =  (s_binary!=0).astype(int)
max_T = s_binary.shape[0] - s_binary.sum()
    
print("Max possible T:", max_T)


cluster_epochs = args.epochs//len(cluster_dict.keys())
bs = args.batch_size
save_folder = os.path.join(args.save_folder, args.data)
save_metrics_folder = os.path.join(args.save_metrics_folder, args.data)
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_metrics_folder, exist_ok=True)


if args.train_gen:
    # train generator (set_embedding + transformer_encoder + decoder)
    loss = ELBO_loss
    optimizer = optim.Adam([{'params' : set_embedding.parameters()}, {'params' : transformer_encoder.parameters()}, {'params' : decoder.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    num_batches = math.ceil(len(train_dataset)/args.batch_size)

    for epoch in range(args.epochs):
        total_loss = 0
        for i in range(num_batches):
            optimizer.zero_grad()
            X,Y = train_dataset[i*bs:(i+1)*bs]
            X = X.values
            X = X/args.norm
            Y = Y.values
            X = torch.tensor(X).to(device)
            Y = torch.tensor(Y).to(device)
            present_mask = torch.ones(X.shape)
            mask = torch.bernoulli(torch.ones(X.shape[1])*args.p).unsqueeze(0).expand(X.shape)*present_mask
            s = set_embedding(X, mask)
            z, mean, logvar = VAE_sample(transformer_encoder, s, args.latent_dim)
            out = decoder(z)
            batch_loss = loss(out, X, present_mask, mean, logvar, args)
            total_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
        if (epoch+1)%args.save_freq == 0:
            torch.save(set_embedding.state_dict(), os.path.join(save_folder, f'set_embedding_{epoch}'))
            torch.save(transformer_encoder.state_dict(), os.path.join(save_folder, f'encoder_{epoch}'))
            torch.save(decoder.state_dict(), os.path.join(save_folder, f'decoder_{epoch}'))
        rmse = torch.sqrt(torch.mean((out-X)**2))
        print(f"Epoch {epoch} done, Loss {total_loss}, RMSE {rmse}")
    print("Generator Trained")
else:
    if args.data=='mnist':
        saved_num = 199
    elif args.data=='disease_pred':
        saved_num = 99
    set_embedding.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_{saved_num}'), map_location=device))
    transformer_encoder.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_{saved_num}'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_{saved_num}'), map_location=device))

# Classify
c = args.cluster_id

if args.train_vanilla:
    loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_batches = math.ceil(len(train_dataset)/args.batch_size)
    for epoch in range(args.epochs):
        total_loss = 0
        total = 0
        correct = 0
        transformer_encoder.train()
        set_embedding.train()
        classifier.train()
        for i in range(num_batches):
            optimizer.zero_grad()
            X,Y = train_dataset[i*bs:(i+1)*bs]
            X = X.values
            X = X/args.norm
            Y = Y.values
            X = torch.tensor(X).to(device)
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = torch.ones_like(X)
            optimizer.zero_grad()
            encoded, _ = transformer_encoder(set_embedding(X, mask_S))
            out = classifier(encoded)
            batch_loss = loss(out, Y)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss

        with torch.no_grad():
            transformer_encoder.eval()
            set_embedding.eval()
            classifier.eval()
            for i in range(num_batches):
                X,Y = train_dataset[i*bs:(i+1)*bs]
                X = X.values
                X = X/args.norm
                Y = Y.values
                X = torch.tensor(X).to(device)
                Y = torch.tensor(Y).to(torch.int64).to(device)
                mask_S = torch.ones_like(X)
                encoded, _ = transformer_encoder(set_embedding(X, mask_S))
                out = classifier(encoded)
                Y_hat = torch.argmax(out, dim=1)
                correct += torch.sum(Y==Y_hat)
                total += Y.shape[0]

        acc = correct/total
        print(f"Vanilla Classifier epoch {epoch} total loss {total_loss} acc {acc}")
    path = os.path.join(save_folder, f'classifier_c{c}_ns{args.num_sets}_vanilla')
    torch.save(classifier.state_dict(), path)
    exit()

if args.test_vanilla:
    total = 0
    correct = 0
    classifier.load_state_dict(torch.load(os.path.join(save_folder, f'classifier_c{c}_ns{args.num_sets}_vanilla')))
    num_batches = math.ceil(len(train_dataset)/args.batch_size)
    with torch.no_grad():
        for i in range(num_batches):
            X,Y = train_dataset[i*bs:(i+1)*bs]
            X = X.values
            X = X/args.norm
            Y = Y.values
            X = torch.tensor(X).to(device)
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = torch.ones_like(X)
            encoded, _ = transformer_encoder(set_embedding(X, mask_S))
            out = classifier(encoded)
            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(Y==Y_hat)
            total += Y.shape[0]
    print("Vanilla acc : ", correct/total)
    exit()

classifier.train()
data, labels, sets = cluster_dict[c]["samples"], cluster_dict[c]["labels"], cluster_dict[c]["sets"]
num_batches = math.ceil(len(data)/args.batch_size)
T = []
data_std_dev = torch.std(torch.tensor(data), axis=0)

loss = nn.CrossEntropyLoss(reduction='none').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction='none')

total_cost = 0
epoch=0
costs = get_costs(args.data)[0]
masks = torch.stack([set_to_mask(s, args.set_size) for s in sets])
costs = torch.Tensor(costs)
# print("Costs:", len(costs))
num_bad_epochs = 0

masks = torch.stack([set_to_mask(s, args.set_size) for s in sets])
ft = True

if args.train_x_S_only or args.warm_start:

    if args.train_random:
        # not checked for bugs
        set_size = data[0].shape[0]
        mask = torch.stack([set_to_mask(s, set_size) for s in sets]) # shape [batch_size, set_size]
        mask_T = set_to_mask(T, set_size)
        complement_mask = 1-torch.gt(mask.sum(dim=0)+mask_T,0).int()
        available_indices = torch.nonzero(complement_mask, as_tuple=True)[0]
        ran_idx = torch.randint(len(available_indices), (1,))
        T = available_indices[ran_idx]
        print("Random T", T)
    elif args.train_fixed_T:
        # not checked for bugs
        T = [88, 97, 23, 108, 59, 61, 124, 57, 102, 26, 0, 121, 17, 3, 103, 55, 64, 67, 32, 84, 130, 70, 16]
        print("Fixed T", T)
    loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.warm_start_lr if args.warm_start else args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        total = 0
        correct = 0
        for i in range(num_batches):
            X = data[i*bs:(i+1)*bs]  
            # data_std_dev = torch.std(X, axis=0)
            X = X/args.norm
            X = torch.tensor(X).to(device)
            Y = labels[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device)
            # S = sets[i*bs:(i+1)*bs]
            mask_S = masks[i*bs:(i+1)*bs]
            if args.train_random or args.train_fixed_T:
                mask_S[:, T] = 1
            optimizer.zero_grad()
            encoded, _ = transformer_encoder(set_embedding(X, mask_S))
            out = classifier(encoded)
            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(Y==Y_hat)
            total += Y.shape[0]
            batch_loss = loss(out, Y)
            batch_loss.backward()
            optimizer.step()
        acc = correct/total
        print(f"Epoch {epoch} accuracy {acc}")
    if(not args.warm_start):
        exit()
    # exit()
# loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
print("Loss:" ,loss)
loss = nn.CrossEntropyLoss(reduction='none').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction='none')


prev_acc = 0
breakout = False
epoch = 0

if(args.end_to_end):
    # optimizer = optim.Adam(list(set_embedding.parameters()) +  list(transformer_encoder.parameters()) + list(decoder.parameters()) +list(classifier.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.Adam([{'params' : set_embedding.parameters()}, {'params' : transformer_encoder.parameters()}, {'params' : decoder.parameters()}, {'params' : classifier.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    print("Training end to end")
else:
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Not Training end to end")

greedy_done = False
T = []
print("DATA SIZE:", len(data))
# for epoch in range(cluster_epochs):
topk = 5
epoch_max = 1e7
max_T = 160
while epoch<epoch_max and not args.no_greedy: # WARNING: hard coded epochs
    epoch_loss = 0
    epoch+=1
    ugf = 0
    transformer_encoder.train()
    set_embedding.train()
    classifier.train()
    for i in range(num_batches):
        X = data[i*bs:(i+1)*bs]  
        X = X/args.norm
        X = torch.tensor(X).to(device)
        Y = labels[i*bs:(i+1)*bs]
        Y = torch.tensor(Y).to(torch.int64).to(device)
        mask_S = masks[i*bs:(i+1)*bs]
        optimizer.zero_grad()
        mask_new = copy.deepcopy(mask_S)
        mask_new[:, T] = 1
        X_new = X
        X_new = torch.tensor(X_new).to(device)
        encoded, _ = transformer_encoder(set_embedding(X, mask_S))
        out = classifier(encoded)
        ug, ug_vec, out_new, out_g = calculate_loss(transformer_encoder, set_embedding, classifier, decoder, VAE_sample, X_new, mask_S, mask_new, args, data_std_dev)
        u = 1 - torch.max(out)
        batch_loss = torch.sum(u*((ug)*loss(out_new, Y) + (1-ug)*loss(out_g, Y)) + (1-u)*loss(out, Y))
        reduced_loss = torch.sum((ug)*loss(out_new, Y) + (1-ug)*loss(out_g, Y))
        epoch_loss+=reduced_loss
        batch_loss.backward()
        optimizer.step()
    total = 0
    correct = 0
    with torch.no_grad():
        transformer_encoder.eval()
        set_embedding.eval()
        classifier.eval()
        for i in range(num_batches):
            X = data[i*bs:(i+1)*bs]  
            X = X/args.norm
            X = torch.tensor(X).to(device)
            Y = labels[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = masks[i*bs:(i+1)*bs]
            mask_new = copy.deepcopy(mask_S)
            mask_new[:, T] = 1
            encoded, _ = transformer_encoder(set_embedding(X, mask_new))
            out = classifier(encoded)
            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(Y==Y_hat)
            total += Y.shape[0]
        acc = correct/total
    if (acc == prev_acc and args.train_breakout):
        breakout = True
        T = [i for i in range(args.set_size)]
        print("Breakout")
    print(f"Epoch {epoch} accuracy {acc}")
    prev_acc = acc
    path = os.path.join(save_folder, f'classifier__LSH_c{c}_ns{args.num_sets}_{epoch+1}_sizeofT{len(T)}')
    path_T = os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{epoch+1}_sizeofT{len(T)}')
    with open(path_T, "wb") as f:
        pickle.dump(T, f)
    torch.save(classifier.state_dict(), path)
    if (epoch+1)%args.greedy_freq==0 and not breakout and not greedy_done:
        start_time = time.time()
        new_feature = greedy(transformer_encoder, decoder, classifier, set_embedding, VAE_sample, loss, data, labels, sets, T, epoch_loss, device, args, topk=topk)
        if new_feature==None:
            num_bad_epochs += 1
            print("Bad epoch, continuing")
            if len(T)>=max_T or (epoch>epoch_max): # WARNING : hard coded max T just before greedy
                break
            continue
        end_time = time.time()
        ugf = ugf/num_batches
        #if (total_cost + (ugf*costs)[new_feature] > args.budget) or (epoch>=args.epochs):
        if len(T)>=max_T or (epoch>epoch_max):
            break
        if topk == 1:
            T.append(new_feature)
            
            total_cost+=costs[new_feature]
            
            print(f'Added feature {new_feature} | Length of T is {len(T)} | Now T is {T} | Total cost = {total_cost} | Time taken = {end_time-start_time}')
        else:
            T += new_feature
            total_cost += torch.sum(costs[new_feature])
            print(f'Added feature {new_feature} | Length of T is {len(T)} | Now T is {T} | Total cost = {total_cost} | Time taken = {end_time-start_time}')


print("Greedy done")

T_g = None
ug = args.ug


if args.warm_end:
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()

    classifier_path = f"classifier__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5 + 2}_sizeofT{args.warm_end_load_T}"
    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5+2}_sizeofT{args.warm_end_load_T}'), "rb"))

    print(classifier_path, T)

    
    if args.warm_end_loss == 'generative_loss' and args.generate_during_test:
        T_g = (torch.tensor(T)[torch.bernoulli(torch.ones(len(T))*ug).bool()]).tolist()
        print("Samples from generator will be obtained for", T_g)

    if not args.warm_end_fresh_start:
        classifier.load_state_dict(torch.load(os.path.join(save_folder, classifier_path)))

    if args.warm_end_loss == 'cross_entropy':
        loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    elif args.warm_end_loss == 'generative_loss':
        loss = nn.CrossEntropyLoss(reduction='none').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction='none')
    
    if args.warm_end_end_to_end:
        optimizer = optim.Adam([{'params' : set_embedding.parameters()}, {'params' : transformer_encoder.parameters()}, {'params' : decoder.parameters()}, {'params' : classifier.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
        print("Training end to end")
    else:
        optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("Not Training end to end")
    
    acc_last = -10
    for epoch in range(args.epochs):
        if args.warm_end_end_to_end:
            classifier.train()
            set_embedding.train()
            transformer_encoder.train()
            decoder.train()
        else:
            classifier.train()
            set_embedding.eval()
            transformer_encoder.eval()
            decoder.eval()

        for i in range(num_batches):
            X = data[i*bs:(i+1)*bs]  
            X = X/args.norm
            X = torch.tensor(X).to(device)
            Y = labels[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = masks[i*bs:(i+1)*bs]
            mask_new = copy.deepcopy(mask_S)
            mask_new[:, T] = 1
            optimizer.zero_grad()
            if args.warm_end_loss == 'cross_entropy':
                encoded, _ = transformer_encoder(set_embedding(X, mask_new))
                out = classifier(encoded)
                batch_loss = loss(out, Y)
            elif args.warm_end_loss == 'generative_loss':
                X_new = X
                X_new = torch.tensor(X_new).to(device)
                encoded, _ = transformer_encoder(set_embedding(X, mask_S))
                out = classifier(encoded)
                ug, ug_vec, out_new, out_g = calculate_loss(transformer_encoder, set_embedding, classifier, decoder, VAE_sample, X_new, mask_S, mask_new, args, data_std_dev)
                u = 1 - torch.max(out)
                batch_loss = torch.sum(u*((ug)*loss(out_new, Y) + (1-ug)*loss(out_g, Y)) + (1-u)*loss(out, Y))
            batch_loss.backward()
            optimizer.step()

        with torch.no_grad():
            total = 0
            correct = 0
            
            classifier.eval()
            set_embedding.eval()
            transformer_encoder.eval()
            decoder.eval()
            for i in range(num_batches):
                X = data[i*bs:(i+1)*bs]  
                X = X/args.norm
                X = torch.tensor(X).to(device)
                Y = labels[i*bs:(i+1)*bs]
                Y = torch.tensor(Y).to(torch.int64).to(device)
                mask_S = masks[i*bs:(i+1)*bs]
                mask_new = copy.deepcopy(mask_S)
                mask_new[:, T] = 1
                if args.warm_end_loss == 'generative_loss' and args.generate_during_test:
                    # note that with generative loss, training was happening with full queried T, but accuracy is with generation+querying
                    z, _, _ = VAE_sample(transformer_encoder, set_embedding(X, mask_S), args.latent_dim)
                    X_g = decoder(z)
                    mask_g = torch.zeros_like(X_g)
                    mask_g[:, T_g] = 1
                    X_new = X*(1-mask_g) + X_g*mask_g
                    encoded, _ = transformer_encoder(set_embedding(X_new, mask_new))
                    out = classifier(encoded)
                else:
                    encoded, _ = transformer_encoder(set_embedding(X, mask_new))
                    out = classifier(encoded)
                Y_hat = torch.argmax(out, dim=1)
                correct += torch.sum(Y==Y_hat)
                total += Y.shape[0]
            acc = correct/total
            if acc > acc_last:
                classifier_final =  copy.deepcopy(classifier)
                set_embedding_final =  copy.deepcopy(set_embedding)
                encoder_final =  copy.deepcopy(transformer_encoder)
                acc_last = acc
            print(f"Warm end : Epoch {epoch} accuracy {acc}")

    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}_warm_end_fresh_start_{args.warm_end_fresh_start}'

    classifier =  copy.deepcopy(classifier_final)
    set_embedding =  copy.deepcopy(set_embedding_final)
    transformer_encoder =  copy.deepcopy(encoder_final)
    

    torch.save(classifier.state_dict(), os.path.join(save_folder, f'classifier__LSH_c{c}' + suffix))
    torch.save(set_embedding.state_dict(), os.path.join(save_folder, f'set_embedding_99_LSH_c{c}' + suffix))
    torch.save(transformer_encoder.state_dict(), os.path.join(save_folder, f'encoder_99_LSH_c{c}' + suffix))
    torch.save(decoder.state_dict(), os.path.join(save_folder, f'decoder_99_LSH_c{c}' + suffix))

    path_T = os.path.join(save_folder, f'T_LSH_c{c}' + suffix)
    pickle.dump(T, open(path_T, 'wb'))


    print("DATA SIZE", len(data))
    print("FINAL ACC", acc.item())
    print("DATA SIZE*ACC", len(data)*acc.item())

    pickle.dump(acc.item(), open(os.path.join(save_metrics_folder, f"train_acc_LSH__c{c}_{args.warm_end_load_T}__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__generate_during_test_{args.generate_during_test}__warm_end_fresh_start_{args.warm_end_fresh_start}.pkl"), "wb"))



if args.test_on_train:


    # Not updated for generative loss
    with torch.no_grad():
        total = 0
        correct = 0
        transformer_encoder.eval()
        set_embedding.eval()
        classifier.eval()
        for i in range(num_batches):
            X = data[i*bs:(i+1)*bs]  
            X = X/args.norm
            X = torch.tensor(X).to(device)
            Y = labels[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = masks[i*bs:(i+1)*bs]
            
            mask_new = copy.deepcopy(mask_S)
            mask_new[:, T] = 1
            # X_new = get_samples_with_features_batch(S_new, indices[i*bs:(i+1)*bs], train_dataset)
            # mask_new = [set_to_mask(s, args.set_size) for s in S_new]
            encoded, _ = transformer_encoder(set_embedding(X, mask_new))
            out = classifier(encoded)
            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(Y==Y_hat)
            total += Y.shape[0]
        acc = correct/total
    print("acc:", acc)


if(args.test):
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()
        
    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T-10}_warm_end_fresh_start_{args.warm_end_fresh_start}'

    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5+2}_sizeofT{args.warm_end_load_T}'), "rb"))
   

    transformer_encoder.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_99_LSH_c{c}' + suffix), map_location=device))
    set_embedding.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_99_LSH_c{c}' + suffix), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_99_LSH_c{c}' + suffix), map_location=device))
    classifier.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__LSH_c{c}' + suffix), map_location=device))
    val_all_labels = torch.stack(val_all_labels)

    val_all_labels = val_all_labels.cpu().numpy()
    val_all_samples = val_all_samples.cpu().numpy()
    print("test data size", val_all_samples.shape, val_all_labels.shape, val_cluster_labels.shape)
    cluster_dict = {}
    for l in np.unique(val_cluster_labels):
        sets = []
        ids = (val_cluster_labels==l)
        for i in np.arange(len(val_all_sets))[ids]:
            sets.append(val_all_sets[i])
        cluster_dict[l] = {
            "samples": val_all_samples[ids], 
            "labels": val_all_labels[ids],
            "sets": sets
            }
    
    data, labels, sets = cluster_dict[c]["samples"], cluster_dict[c]["labels"], cluster_dict[c]["sets"]
    with open(f"data/test_samples_{c}", "wb") as f:
        pickle.dump((data,labels, sets), f)
    print(len(data), c)
    total = 0
    correct = 0
    Q_plus = 0
    Q_minus = 0
    masks = torch.stack([set_to_mask(s, args.set_size) for s in sets])
    num_batches = math.ceil(len(data)/args.batch_size)
    S_list = torch.nonzero(torch.sum(masks, dim=0), as_tuple=True)[0].tolist()
    T_val = [t for t in T if t not in S_list]
    # T_val = T
    print("val T", T_val)

    if not T_g and args.generate_during_test:
        T_g = (torch.tensor(T_val)[torch.bernoulli(torch.ones(len(T_val))*ug).bool()]).tolist()


    print("Samples from generator will be obtained for", T_g)
    
    with torch.no_grad():
        transformer_encoder.eval()
        set_embedding.eval()
        classifier.eval()
        decoder.eval()
        accuracies = [0]
        sizes = [0]
        bucket_size = int(math.ceil(len(data)/10))
        for i in range(num_batches):
            X = data[i*bs:(i+1)*bs]  
            X = X/args.norm
            X = torch.tensor(X).to(device)
            Y = labels[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = masks[i*bs:(i+1)*bs]
            
            mask_new = copy.deepcopy(mask_S)
            mask_new[:, T_val] = 1
            
            #UWA
            encoded_S, _ = transformer_encoder(set_embedding(X, mask_S))
            out_S = classifier(encoded_S)
            Y_hat_S = torch.argmax(out_S, dim=1)

            if args.generate_during_test:
                z, _, _ = VAE_sample(transformer_encoder, set_embedding(X, mask_S), args.latent_dim)
                X_g = decoder(z)
                mask_g = torch.zeros_like(X_g)
                mask_g[:, T_g] = 1
                X_new = X*(1-mask_g) + X_g*mask_g
                encoded, _ = transformer_encoder(set_embedding(X_new, mask_new))
                out = classifier(encoded)
            else:
                encoded, _ = transformer_encoder(set_embedding(X, mask_new))
                out = classifier(encoded)

            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(Y==Y_hat)
            total += Y.shape[0]
            accuracies[-1] += torch.sum(Y==Y_hat)
            sizes[-1] += Y.shape[0]
            if sizes[-1] > bucket_size:
                accuracies[-1] = accuracies[-1]/sizes[-1]
                accuracies.append(0)
                sizes.append(0)
            # Q_plus += torch.sum((Y==Y_hat).int() * (~(Y==Y_hat_S)).int())
            # Q_minus += torch.sum(Y==Y_hat_S)

    acc = correct/total
    print(accuracies[:-1], sizes[:-1])
    print("TEST DATA SIZE", len(data))
    print("TEST ACC", acc)
    print("TEST Q_plus", Q_plus/total)
    print("Test Q_minus", Q_minus/total)

    metrics_dict = {
        "args": args,
        "data size": len(data),
        "acc": acc,
        "Q_plus": Q_plus,
        "Q_minus" : Q_minus,
        "Cost": len(T_val)-len(T_g) if T_g else len(T_val),
    }
    with open(os.path.join(save_metrics_folder, f"jafa__c_{c}__T_{args.warm_end_load_T}__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__generate_during_test_{args.generate_during_test}__generate_for_min_{args.generate_for_min}.pkl"), "wb") as f:
        pickle.dump(metrics_dict, f)
