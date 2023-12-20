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
import matplotlib.pyplot as plt
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
  
sys.path.append(parent_directory)

from data import datasets
from data.costs import get_costs
from utils import *

from models.transformer import Transformer_Encoder, Transformer_Encoder_pytorch, Set_Embedding, Configuration, VAE_sample, ELBO_loss
from models.decoder import Decoder
from models.classifier import Classifier, ChoiceClassifier

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
parser.add_argument('--greedy_sample_size', type=int, default=6000)
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
parser.add_argument('--test_with_choice', action='store_true', default=False)
parser.add_argument('--choice_encoder_generative_loss', action='store_true', default=False)
parser.add_argument('--choice_threshold', default=0.5, type=float)
parser.add_argument('--choice_frac', default=0.1, type=float)
parser.add_argument('--gen_small',default=False, action='store_true')
parser.add_argument('--metrics', default=False, action='store_true')

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
    new_val_dataset = datasets.MnistDataset("data/mnist/Validation.csv")
    args.norm = 256.0
elif args.data=='disease_pred':
    train_dataset = datasets.DiseasePredDataset("data/disease_pred/Training.csv")
    val_dataset = datasets.DiseasePredDataset("data/disease_pred/Testing.csv")
    new_val_dataset = datasets.DiseasePredDataset("data/disease_pred/Validation.csv")
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
        
        # samples = samples[non_zero_indices,:]
        # masked_samples = masked_samples[non_zero_indices,:]
        # masked_samples[zero_indices] = masked_samples[zero_indices]
        # labels = labels[non_zero_indices]
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


    new_val_all_samples = []
    new_val_all_labels = []
    new_val_all_sets = []
    new_val_all_masked_samples = []

    for s in val_initial_sets:
        masked_samples, samples, labels = get_samples_with_features(s, new_val_dataset)
        non_zero_indices = torch.where(masked_samples.sum(axis = 1) != 0)[0]
        
        # samples = samples[non_zero_indices,:]
        # masked_samples = masked_samples[non_zero_indices,:]
        # masked_samples[zero_indices] = masked_samples[zero_indices]
        # labels = labels[non_zero_indices]
        new_val_all_samples+=samples[non_zero_indices]
        new_val_all_labels+=labels[non_zero_indices]
        new_val_all_sets+=([s]*len(non_zero_indices))
        new_val_all_masked_samples+=masked_samples[non_zero_indices]

    new_val_all_samples = torch.stack(new_val_all_samples)
    new_val_all_masked_samples = torch.stack(new_val_all_masked_samples)
    new_val_cluster_labels = []
    for s in new_val_all_masked_samples:
        new_val_cluster_labels.append(val_cluster_obj.retrieve(s.cpu()))

    new_val_cluster_labels = np.array(new_val_cluster_labels)


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
    transformer_encoder.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_{saved_num}'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_{saved_num}'), map_location=device))
    set_embedding.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_{saved_num}'), map_location=device))

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
# print(data[:5])
data_std_dev = torch.std(torch.tensor(data), axis=0)
# print(f'Data Std Dev = {data_std_dev}')

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
while epoch<300 and not args.no_greedy:
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
        new_feature = greedy(transformer_encoder, decoder, classifier, set_embedding, VAE_sample, loss, data, labels, sets, T, epoch_loss, device, args)
        if new_feature==None:
            num_bad_epochs += 1
            print("Bad epoch, continuing")
            continue
        end_time = time.time()
        ugf = ugf/num_batches
        #if (total_cost + (ugf*costs)[new_feature] > args.budget) or (epoch>=args.epochs):
        if len(T)==max_T or (epoch>300):
            break
        T.append(new_feature)
        
        total_cost+=costs[new_feature]
        
        print(f'Added feature {new_feature} | Length of T is {len(T)} | Now T is {T} | Total cost = {total_cost} | Time taken = {end_time-start_time}')

print("Greedy done")
T_g = None
ug = args.ug

if args.gen_small:
    loss = ELBO_loss
    transformer_encoder_small = Transformer_Encoder_pytorch(args)
    set_embedding_small = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
    decoder_small = Decoder(args.latent_dim, args.set_size)
    # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)
    classifier_small = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
    transformer_encoder_small = transformer_encoder_small.to(device)
    set_embedding_small = set_embedding_small.to(device)
    decoder_small = decoder_small.to(device)
    optimizer = optim.Adam([{'params' : set_embedding_small.parameters()}, {'params' : transformer_encoder_small.parameters()}, {'params' : decoder_small.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
    num_batches = math.ceil(len(data)/args.batch_size)

    for epoch in range(args.epochs):
        total_loss = 0
        for i in range(num_batches):
            optimizer.zero_grad()
            X = data[i*bs:(i+1)*bs]  
            X = X/args.norm
            X = torch.tensor(X).to(device)
            Y = labels[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device)
            mask_S = masks[i*bs:(i+1)*bs]
            # indices = torch.randperm()
            mask_new = copy.deepcopy(mask_S)
            mask_new[:,T] = 1
            present_mask = torch.ones_like(mask_new)
            z, mean, logvar = VAE_sample(transformer_encoder_small, set_embedding_small(X, mask_S), args.latent_dim)
            out = decoder_small(z)
            batch_loss = loss(out*mask_new, X*mask_new, present_mask, mean, logvar, args)
            total_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            if i==0:
                images = out.cpu().detach().numpy().reshape((-1,28,28))
                for j in range(5):
                    plt.imshow(images[j])
                    plt.savefig(f"images/image_cluster_{c}_epoch_{epoch}_{j}")
        if (epoch+1)%args.save_freq == 0:
            torch.save(set_embedding_small.state_dict(), os.path.join(save_folder, f'set_embedding_small_{c}_{epoch}'))
            torch.save(transformer_encoder_small.state_dict(), os.path.join(save_folder, f'encoder_small_{c}_{epoch}'))
            torch.save(decoder_small.state_dict(), os.path.join(save_folder, f'decoder_small_{c}_{epoch}'))
        print(f"Epoch {epoch} done, Loss {total_loss}")



if args.warm_end:
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()
    classifier_path = f"classifier__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5 + 2}_sizeofT{args.warm_end_load_T}"
    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5+2}_sizeofT{args.warm_end_load_T}'), "rb"))
    
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

    transformer_encoder_gen = Transformer_Encoder_pytorch(args)
    set_embedding_gen = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
    decoder_gen = Decoder(args.latent_dim, args.set_size)
    # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)
    classifier_gen = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
    transformer_encoder_gen = transformer_encoder_gen.to(device)
    set_embedding_gen = set_embedding_gen.to(device)
    decoder_gen = decoder_gen.to(device)

    suffix = f'_ns40_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_40'
    transformer_encoder_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_encoder_99_LSH_c{c}'+suffix)))
    set_embedding_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_set_embedding_99_LSH_c{c}'+suffix)))
    decoder_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_decoder_99_LSH_c{c}'+suffix)))

    transformer_encoder_gen.eval()
    set_embedding_gen.eval()
    decoder_gen.eval()
    
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
            # mask_new = torch.ones_like(mask_new)
            optimizer.zero_grad()
            z, _, _ = VAE_sample(transformer_encoder_gen, set_embedding_gen(X, mask_S), args.latent_dim)
            X_g = decoder_gen(z)
            X_combined = X*mask_S + X_g*(1-mask_S)
            encoded, _ = transformer_encoder(set_embedding(X_combined, mask_new))
            out = classifier(encoded)
            batch_loss = loss(out, Y)
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
                # mask_new = torch.ones_like(mask_new)
                z, _, _ = VAE_sample(transformer_encoder_gen, set_embedding_gen(X, mask_S), args.latent_dim)
                X_g = decoder_gen(z)
                X_combined = X*mask_S + X_g*(1-mask_S)
                encoded, _ = transformer_encoder(set_embedding(X_combined, mask_new))
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

    ls = [i for i in range(10)]
    with torch.no_grad():
        total_Y = [0 for i in range(10)]
        correct_Y = [0 for i in range(10)]
        total_Y_hat = [0 for i in range(10)]
        correct_Y_hat = [0 for i in range(10)]
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
            # mask_new = torch.ones_like(mask_new)
            z, _, _ = VAE_sample(transformer_encoder_gen, set_embedding_gen(X, mask_S), args.latent_dim)
            X_g = decoder_gen(z)
            X_combined = X*mask_S + X_g*(1-mask_S)
            encoded, _ = transformer_encoder(set_embedding(X_combined, mask_new))
            out = classifier(encoded)
            Y_hat = torch.argmax(out, dim=1)
            total += Y.shape[0]
            for l in ls:
                total_Y[l] += torch.sum(Y==l)
                total_Y_hat[l] += torch.sum(Y_hat==l)
                correct_Y[l] += torch.sum(torch.logical_and(Y==l,Y==Y_hat))
                correct_Y_hat[l] += torch.sum(torch.logical_and(Y_hat==l,Y==Y_hat))

    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}'

    classifier =  copy.deepcopy(classifier_final)
    set_embedding =  copy.deepcopy(set_embedding_final)
    transformer_encoder =  copy.deepcopy(encoder_final)
    

    torch.save(classifier.state_dict(), os.path.join(save_folder, f'classifier__LSH_A1_c{c}' + suffix))
    torch.save(set_embedding.state_dict(), os.path.join(save_folder, f'set_embedding_99_LSH_A1_c{c}' + suffix))
    torch.save(transformer_encoder.state_dict(), os.path.join(save_folder, f'encoder_99_LSH_A1_c{c}' + suffix))
    torch.save(decoder.state_dict(), os.path.join(save_folder, f'decoder_99_LSH_A1_c{c}' + suffix))

    path_T = os.path.join(save_folder, f'T_LSH_c{c}' + suffix)
    pickle.dump(T, open(path_T, 'wb'))

    with open(f"hist_{c}.pkl", "wb") as f:
        pickle.dump({
            "total_Y" : total_Y,
            "total_Y_hat" : total_Y_hat,
            "correct_Y" : correct_Y,
            "correct_Y_hat" : correct_Y_hat
        } ,f)


    # print("DATA SIZE", len(data))
    # print("FINAL ACC", acc.item())
    # print("DATA SIZE*ACC", len(data)*acc.item())

    # pickle.dump(acc.item(), open(os.path.join(save_metrics_folder, f"train_acc_LSH__c{c}_{args.warm_end_load_T}__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__generate_during_test_{args.generate_during_test}.pkl"), "wb"))



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

if args.test_with_choice:
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()
    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}'
    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5+2}_sizeofT{args.warm_end_load_T}'), "rb"))

    classifier.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__LSH_A1_c{c}' + suffix)))
    if args.choice_encoder_generative_loss:
        suffix_new = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_generative_loss__T_{args.warm_end_load_T}'
        transformer_encoder.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_99_LSH_c{c}' + suffix_new)))
        set_embedding.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_99_LSH_c{c}' + suffix_new)))
        decoder.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_99_LSH_c{c}' + suffix_new)))
    else:
        suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}'
        transformer_encoder.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_99_LSH_A1_c{c}' + suffix)))
        set_embedding.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_99_LSH_A1_c{c}' + suffix)))
        decoder.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_99_LSH_A1_c{c}' + suffix)))
        
        transformer_encoder_gen = Transformer_Encoder_pytorch(args)
        set_embedding_gen = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
        decoder_gen = Decoder(args.latent_dim, args.set_size)
        # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)
        classifier_gen = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
        transformer_encoder_gen = transformer_encoder_gen.to(device)
        set_embedding_gen = set_embedding_gen.to(device)
        decoder_gen = decoder_gen.to(device)

        suffix = f'_ns40_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_40'
        transformer_encoder_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_encoder_99_LSH_c{c}'+suffix)))
        set_embedding_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_set_embedding_99_LSH_c{c}'+suffix)))
        decoder_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_decoder_99_LSH_c{c}'+suffix)))
        transformer_encoder_gen.eval()
        set_embedding_gen.eval()
        decoder_gen.eval()

    num_batches = math.ceil(len(data)/args.batch_size)

        
    choice_classifier = ChoiceClassifier(args.encoder_output_dim, args.clf_hidden_sizes, 1, dropout = args.dropout, p = args.clf_p, batch_norm = False).to(device)
    # choice_classifier = Classifier(args.encoder_output_dim, [32, 32], 2, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
    choice_transformer_encoder = Transformer_Encoder_pytorch(args).to(device)
    choice_set_embedding = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim).to(device)
    # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)


    def ce_loss(out, Y):
        y = copy.deepcopy(Y)
        y = 2*y-1
        # print(y.shape, out.shape)
        y = y[:,None]
        mask_plus = y>0
        count_plus = torch.sum(mask_plus.int())
        count_minus = torch.sum((~mask_plus).int())
        return (torch.sum(torch.nn.ReLU()(1-y[mask_plus]*out[mask_plus]))*count_minus + torch.sum(torch.nn.ReLU()(1-y[~mask_plus]*out[~mask_plus]))*count_plus)/(2*count_minus*count_plus)
        # return -torch.sum(torch.log(out[torch.arange(out.shape[0]), y]))

    # loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    loss = ce_loss
    optimizer = optim.Adam([{'params':choice_classifier.parameters()}, {'params': choice_transformer_encoder.parameters()}, {'params' : choice_set_embedding.parameters()}], lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0
    for epoch in range(args.epochs):
        with torch.no_grad():
            transformer_encoder.eval()
            set_embedding.eval()
            classifier.eval()
            decoder.eval()
            data_combined = []
            labels_combined = []
            for i in range(num_batches):
                X = data[i*bs:(i+1)*bs]  
                X = X/args.norm
                X = torch.tensor(X).to(device)
                mask_S = masks[i*bs:(i+1)*bs]
                Y = labels[i*bs:(i+1)*bs]
                Y = torch.tensor(Y).to(torch.int64).to(device)
                z, _, _ = VAE_sample(transformer_encoder_gen, set_embedding_gen(X,  mask_S), args.latent_dim)
                X_g = decoder_gen(z)
                
                X_combined = X*mask_S + X_g*(1-mask_S)
                data_combined.append(X_combined)

                mask_new = copy.deepcopy(mask_S)
                mask_new[:,T] = 1
                # mask_new = torch.ones_like(mask_new)
                encoded, _ = transformer_encoder(set_embedding(X_combined, mask_new))
                out = classifier(encoded)
                Y_hat = torch.argmax(out, dim=1)
                labels_combined.append((Y_hat==Y).int())
            data_combined = torch.cat(data_combined)
            labels_combined = torch.cat(labels_combined)

        choice_classifier.train()
        choice_transformer_encoder.train()
        choice_set_embedding.train()
        for i in range(num_batches):
            optimizer.zero_grad()
            X = data_combined[i*bs:(i+1)*bs]  
            X = X/args.norm
            X = torch.tensor(X).to(device)
            mask_S = masks[i*bs:(i+1)*bs]
            Y = labels_combined[i*bs:(i+1)*bs]
            Y = torch.tensor(Y).to(torch.int64).to(device) 
            mask_new = copy.deepcopy(mask_S)
            mask_new[:,T] = 1
            # mask_new = torch.ones_like(mask_new)
            encoded, _ = choice_transformer_encoder(choice_set_embedding(X, mask_new))
            out = choice_classifier(encoded)
            batch_loss = loss(out, Y)
            batch_loss.backward()
            optimizer.step()

        choice_classifier.eval()
        choice_transformer_encoder.eval()
        choice_set_embedding.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i in range(num_batches):
                X = data_combined[i*bs:(i+1)*bs]  
                X = X/args.norm
                X = torch.tensor(X).to(device)
                mask_S = masks[i*bs:(i+1)*bs]
                Y = labels_combined[i*bs:(i+1)*bs]
                Y = torch.tensor(Y).to(torch.int64).to(device) 
                mask_new = copy.deepcopy(mask_S)
                mask_new[:,T] = 1
                # mask_new = torch.ones_like(mask_new)
                encoded, _ = choice_transformer_encoder(choice_set_embedding(X, mask_new))
                out = choice_classifier(encoded)
                # Y_hat = torch.argmax(out, dim=1)
                Y_hat = torch.gt(out, 0.5).int()
                Y_hat = torch.squeeze(Y_hat)
                assert Y_hat.shape == Y.shape
                correct += torch.sum(Y==Y_hat)
                total += len(Y)
            print(f"Epoch {epoch} acc {correct/total}")
            if correct/total > best_acc:
                best_choice_classifier = copy.deepcopy(choice_classifier)
                best_choice_transformer_encoder = copy.deepcopy(choice_transformer_encoder)
                best_choice_set_embedding = copy.deepcopy(choice_set_embedding)
                best_acc = correct/total


if(args.test):
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()
    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}'
    suffix_new = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}__choice_encoder_generative_loss_{args.choice_encoder_generative_loss}'
    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{2*args.warm_end_load_T//5+2}_sizeofT{args.warm_end_load_T}'), "rb"))

    train_acc = []

    # choice_classifier = Classifier(args.encoder_output_dim, [32,32], 2, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)

    transformer_encoder_query = Transformer_Encoder_pytorch(args)
    set_embedding_query = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
    decoder_query = Decoder(args.latent_dim, args.set_size)
    # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)
    classifier_query = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
    transformer_encoder_query = transformer_encoder_query.to(device)
    set_embedding_query = set_embedding_query.to(device)
    decoder_query = decoder_query.to(device)

    transformer_encoder_sample = Transformer_Encoder_pytorch(args)
    set_embedding_sample = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
    decoder_sample = Decoder(args.latent_dim, args.set_size)
    # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)
    classifier_sample = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
    transformer_encoder_sample = transformer_encoder_sample.to(device)
    set_embedding_sample = set_embedding_sample.to(device)
    decoder_sample = decoder_sample.to(device)

    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}_warm_end_fresh_start_{args.warm_end_fresh_start}'

    transformer_encoder_query.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_99_LSH_c{c}' + suffix), map_location=device))
    set_embedding_query.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_99_LSH_c{c}' + suffix), map_location=device))
    decoder_query.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_99_LSH_c{c}' + suffix), map_location=device))
    classifier_query.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__LSH_c{c}' + suffix), map_location=device))

    classifier_gen = Classifier(args.encoder_output_dim, args.clf_hidden_sizes, args.num_classes, dropout = args.dropout, p = args.clf_p, batch_norm = args.batchnorm).to(device)
    classifier_gen.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__LSH_c{c}' + suffix), map_location=device))

    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}'

    transformer_encoder_query.eval()
    set_embedding_query.eval()
    decoder_query.eval()
    classifier_query.eval()

    transformer_encoder_sample.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_99_LSH_A1_c{c}' + suffix), map_location=device))
    set_embedding_sample.load_state_dict(torch.load(os.path.join(save_folder, f'set_embedding_99_LSH_A1_c{c}' + suffix), map_location=device))
    decoder_sample.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_99_LSH_A1_c{c}' + suffix), map_location=device))
    classifier_sample.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__LSH_A1_c{c}' + suffix), map_location=device))

    transformer_encoder_sample.eval()
    set_embedding_sample.eval()
    decoder_sample.eval()
    classifier_sample.eval()

    transformer_encoder_gen = Transformer_Encoder_pytorch(args)
    set_embedding_gen = Set_Embedding(set_size=args.set_size, hidden_size=args.set_embedding_dim)
    decoder_gen = Decoder(args.latent_dim, args.set_size)
    # classifier = Classifier(args.encoder_output_dim, args.num_classes).to(device)
    transformer_encoder_gen = transformer_encoder_gen.to(device)
    set_embedding_gen = set_embedding_gen.to(device)
    decoder_gen = decoder_gen.to(device)


    # choice_classifier.load_state_dict(torch.load(os.path.join(save_folder, f'choice_classifier__LSH_old_c{c}' + suffix_new)))

    suffix = f'_ns40_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_40'
    transformer_encoder_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_encoder_99_LSH_c{c}' + suffix), map_location=device))
    set_embedding_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_set_embedding_99_LSH_c{c}' + suffix), map_location=device))
    decoder_gen.load_state_dict(torch.load(os.path.join(save_folder, f'old_gen/old_decoder_99_LSH_c{c}' + suffix), map_location=device))

    transformer_encoder_gen.eval()
    set_embedding_gen.eval()
    decoder_gen.eval()

    with open(f"data/test_samples_{c}", "rb") as f:
        full_data, full_labels, full_sets = pickle.load(f)

    full_masks = torch.stack([set_to_mask(s, args.set_size) for s in full_sets])
    S_list = torch.nonzero(torch.sum(full_masks, dim=0), as_tuple=True)[0].tolist()
    T_val = [t for t in T if t not in S_list]
    # T_val = T
    print("val T", T_val)

    full_len = len(full_data)
    perm = np.random.permutation(full_len)
    full_data = full_data[perm]
    full_labels = full_labels[perm]
    full_masks = full_masks[perm]

    num_mc = 1
    mc_bs = math.ceil(full_len/num_mc)
    results = []
    for mc in range(num_mc):
        data = full_data[mc*mc_bs:(mc+1)*mc_bs]
        labels = full_labels[mc*mc_bs:(mc+1)*mc_bs]
        masks = full_masks[mc*mc_bs:(mc+1)*mc_bs]

        print(len(data), c)
        total = 0
        correct = 0
        Q_plus = 0
        Q_minus = 0
        total_cost = 0
        num_batches = math.ceil(len(data)/args.batch_size)
        X_query = []
        X_sample = []
        Y_query = []
        Y_sample = []
        mask_query = []
        mask_sample = []
        ignore_choice = False

        with torch.no_grad():
            C_list = []
            Y_list = []
            num_batches = math.ceil(len(data)/args.batch_size)
            for i in range(num_batches):
                X = data[i*bs:(i+1)*bs]  
                X = X/args.norm
                X = torch.tensor(X).to(device)
                Y = labels[i*bs:(i+1)*bs]
                Y = torch.tensor(Y).to(torch.int64).to(device)
                mask_S = masks[i*bs:(i+1)*bs]
                
                mask_new = copy.deepcopy(mask_S)
                mask_new[:, T_val] = 1
                
                #generation
                z, _, _ = VAE_sample(transformer_encoder_gen, set_embedding_gen(X, mask_S), args.latent_dim)
                X_g = decoder_gen(z)

                #choice
                X_combined = X*mask_S + X_g*(1-mask_S)
                encoded0, _ = transformer_encoder_sample(set_embedding_sample(X_combined, mask_new))
                out_g = classifier_sample(encoded0)
                Y_g = torch.argmax(out_g, dim=1)
                confidence = out_g[torch.arange(out_g.shape[0]), Y_g]
                # choice_out = choice_classifier(encoded0)
                # choice_mask = torch.argmax(choice_out, dim=1)
                if ignore_choice: 
                    print("Ignore")
                    choice_mask = torch.zeros(X.shape[0], dtype=int)
                else:
                    csx,perm = torch.sort(confidence, descending=True)
                    k = min(int(args.choice_frac*csx.shape[0]), csx.shape[0])
                    print(k)
                    threshold = copy.deepcopy(csx[k-1])
                    csx[k:] = 0
                    choice_mask = (csx==-1).int()
                    choice_mask[perm] = (csx >= threshold).int()

                choice_mask = torch.gt(choice_mask, 0)
                X_query.append(X[~choice_mask])
                X_sample.append(X_combined[choice_mask])
                Y_query.append(Y[~choice_mask])
                Y_sample.append(Y[choice_mask])
                mask_query.append(mask_S[~choice_mask])
                mask_sample.append(mask_S[choice_mask])

            X_query = torch.cat(X_query)
            X_sample = torch.cat(X_sample)
            Y_query = torch.cat(Y_query)
            Y_sample = torch.cat(Y_sample)
            mask_query = torch.cat(mask_query)
            mask_sample = torch.cat(mask_sample)

            correct = 0
            total = 0
            total_cost = 0

            num_batches = math.ceil(len(X_query)/args.batch_size)
            print("QUERY, sample", X_query.shape, X_sample.shape)
            for i in range(num_batches):
                X = X_query[i*bs:(i+1)*bs]
                Y = Y_query[i*bs:(i+1)*bs]
                mask_S = mask_query[i*bs:(i+1)*bs]
                mask_new = copy.deepcopy(mask_S)
                mask_new[:,T_val] = 1
                encoded, _ = transformer_encoder_query(set_embedding_query(X, mask_new))
                out = classifier_query(encoded)
                Y_hat = torch.argmax(out, dim=1)
                correct += torch.sum(Y==Y_hat)
                total += Y.shape[0]
                total_cost += len(T_val)*len(X)

            num_batches = math.ceil(len(X_sample)/args.batch_size)
            for i in range(num_batches):
                X = X_sample[i*bs:(i+1)*bs]
                Y = Y_sample[i*bs:(i+1)*bs]
                mask_S = mask_sample[i*bs:(i+1)*bs]
                mask_new = copy.deepcopy(mask_S)
                mask_new[:,T_val] = 1
                encoded, _ = transformer_encoder_sample(set_embedding_sample(X, mask_new))
                out = classifier_sample(encoded)
                Y_hat = torch.argmax(out, dim=1)
                correct += torch.sum(Y==Y_hat)
                total += Y.shape[0]
                total_cost += 0


        acc = correct/total
        cost = total_cost/total
        results.append((acc,total))
    print(results)
    print("TEST DATA SIZE", len(data))
    print("TEST ACC", acc)
    print("TEST Q_plus", Q_plus/total)
    print("Test Q_minus", Q_minus/total)
    print("Cost", cost)

    metrics_dict = {
        "args": args,
        "data size": len(data),
        "acc": acc,
        "Q_plus": Q_plus,
        "Q_minus" : Q_minus,
        "Cost": cost,
    }
    with open(os.path.join(save_metrics_folder, f"results_full_T_gen_LSH__c_{c}__T_{args.warm_end_load_T}__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__generate_during_test_{args.generate_during_test}__generate_for_min_{args.generate_for_min}__choice_encoder_generative_loss_{args.choice_encoder_generative_loss}__choice_frac_{args.choice_frac}.pkl"), "wb") as f:
        pickle.dump(metrics_dict, f)