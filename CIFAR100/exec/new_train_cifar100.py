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
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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
from models.resnet import ResnetEncoder
from models.dcgan import DCGANDecoder
from models.wideresnet import Wide_ResNet

parser = argparse.ArgumentParser()

# cluster related arguments
parser.add_argument('-d', '--data', type=str, default='disease_pred', choices=['disease_pred','mnist','cifar100','tiny_imagenet'])
parser.add_argument('-ns', '--num_sets', type=int, default=20, help='Number of initial sets to consider')
parser.add_argument('-sm', '--sampling_method', type=str, default='fixed_length', choices=['uniform', 'cost_prob', 'fixed_length'], help='sampling method')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-c','--cluster_id', type=int, default=0)

parser.add_argument('--train_gen', action='store_true', default=False)
# parser.add_argument('--set_size', type=int) # size of the universal set of features
parser.add_argument('-nc', '--num_classes', type=int, default=42) # number of classes
parser.add_argument('--p', type=float, default=0.5) # dropout value for generator training
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
parser.add_argument('--encoder_output_dim', type=int, default=100)

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
parser.add_argument('--clf_hidden_sizes', help='clf mlp size',type=str, default='[100,100]')
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
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device(f'cuda:{args.gpu}')
    # torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')
    
print('Using device:',device)

torch.random.manual_seed(0)
np.random.seed(0)

args.set_size = 256

## loading the clusters
with open(f'data/clusters_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    print(f'data/clusters_LSH_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl')
    pkl_data = pickle.load(f)
    train_dataset = pkl_data["train_dataset"]
    all_sets = pkl_data["train_sets"]
    train_indices = pkl_data["train_cluster_indices"]
    val_dataset = pkl_data["test_dataset"]
    val_all_sets = pkl_data["test_sets"]
    val_indices = pkl_data["test_cluster_indices"]

with open(f'data/initial_sets_{args.data}_{args.num_sets}_{args.sampling_method}_s{args.seed}.pkl', 'rb') as f:
    initial_sets = pickle.load(f)

def getT2(T):
  out = []
  for i in T:
    base = 64*(i//16)+2*(i%16)
    out += [base, base+1,base+32,base+33]
  return out

# all_masks = torch.stack([set_to_mask(getT2(s), 1024) for s in all_sets])

bs = args.batch_size
save_folder = os.path.join(args.save_folder, args.data)
save_metrics_folder = os.path.join(args.save_metrics_folder, args.data)
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_metrics_folder, exist_ok=True)

encoder = ResnetEncoder().to(device)
decoder = DCGANDecoder().to(device)
  
# all_masks = torch.stack([set_to_mask(getT2(s), 1024) for s in all_sets])

if args.train_gen:
    # train generator (set_embedding + transformer_encoder + decoder)
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, [i for i in range(len(train_dataset)//len(initial_sets))]), batch_size=bs, shuffle=False)
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        count = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            s = initial_sets[np.random.randint(0,len(initial_sets))]
            mask_S = set_to_mask(getT2(s), 1024) # bs*1024
            mask_S = mask_S.reshape((32,32)).to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            masked_X = (inputs*mask_S[None,None,:,:])
            out = decoder(encoder(masked_X))
            batch_loss = torch.sum((inputs-out)**2)/len(inputs)
            total_loss += batch_loss*len(inputs)
            count += len(inputs)
            batch_loss.backward()
            optimizer.step()
        if (epoch+1)%args.save_freq == 0:
            torch.save(encoder.state_dict(), os.path.join(save_folder, f'encoder_{epoch}'))
            torch.save(decoder.state_dict(), os.path.join(save_folder, f'decoder_{epoch}'))
        rmse = total_loss/count
        print(f"Epoch {epoch} done, train rmse {rmse}")
    print("Generator Trained")
    exit()
else:
    if args.data=='mnist':
        saved_num = 199
    elif args.data=='disease_pred':
        saved_num = 99
    elif args.data == 'cifar100':
        saved_num = 194
    encoder.load_state_dict(torch.load(os.path.join(save_folder, f'encoder_{saved_num}'), map_location=device))
    decoder.load_state_dict(torch.load(os.path.join(save_folder, f'decoder_{saved_num}'), map_location=device))

classifier = Wide_ResNet(28,10,0.3,100).to(device)
c = args.cluster_id

data_subset = torch.utils.data.Subset(train_dataset, train_indices[c])
trainloader = torch.utils.data.DataLoader(data_subset, batch_size=bs, shuffle=False)
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(val_dataset, val_indices[c]), batch_size=bs, shuffle=False)
trainsets = [all_sets[i] for i in train_indices[c]]
testsets = [val_all_sets[i] for i in val_indices[c]]
trainmasks = torch.stack([set_to_mask(getT2(s), 1024) for s in trainsets])
testmasks = torch.stack([set_to_mask(getT2(s), 1024) for s in testsets])
train_size = len(data_subset)

T = []
print("DATA SIZE:", len(train_indices[c]), len(val_indices[c]))

loss = nn.CrossEntropyLoss(reduction='none').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction='none')

costs = get_costs(args.data)[0]
costs = torch.Tensor(costs)


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 60):
        optim_factor = 3
    elif(epoch > 40):
        optim_factor = 2
    elif(epoch > 20):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

if args.warm_start:
    loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        classifier.train()
        classifier.training = True
        train_loss = 0
        total = 0
        correct = 0
        optimizer = optim.SGD(classifier.parameters(), lr=learning_rate(0.1, epoch), momentum=0.9, weight_decay=5e-4)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device) # GPU settings
            mask_S = trainmasks[batch_idx*bs:(1+batch_idx)*bs] # bs*1024
            mask_S = mask_S.reshape((-1,32,32)).to(device)
            # inputs: bs*3*32*32
            optimizer.zero_grad()
            outputs = classifier(inputs*mask_S[:,None,:,:])               # Forward Propagation
            batch_loss = loss(outputs, targets)  # Loss
            batch_loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update

            train_loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                    %(epoch, args.epochs, batch_idx+1,
                        (len(data_subset)//bs)+1, batch_loss.item(), 100.*correct/total))
            sys.stdout.flush()

        with torch.no_grad():
            classifier.eval()
            classifier.training = False
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    mask_S = testmasks[batch_idx*bs:(1+batch_idx)*bs]
                    mask_S = mask_S.reshape((-1,32,32)).to(device)
                    outputs = classifier(inputs*mask_S[:,None,:,:])
                    batch_loss = loss(outputs, targets)

                    test_loss += batch_loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                # Save checkpoint when best model
                acc = 100.*correct/total
                print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, batch_loss.item(), acc))

        path = os.path.join(save_folder, f'classifier__warm_start_c{c}_ns{args.num_sets}')
        torch.save(classifier.state_dict(), path)

classifier.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__warm_start_c{c}_ns{args.num_sets}'), map_location=device))
classifier = classifier.to(device)
loss = nn.CrossEntropyLoss(reduction='none').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction='none')
if(args.end_to_end):
    optimizer = optim.Adam(list(classifier.parameters())+list(decoder.parameters())+list(encoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    print("Training end to end")
else:
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Not Training end to end")


greedy_done = False
T = []

topk = 10
epoch_max = 1e7
mask_util = torch.stack([set_to_mask(s,256) for s in trainsets]) # bs*256
complement_mask_util= 1-torch.gt(mask_util.sum(dim=0),0).int()
max_T = torch.sum(complement_mask_util)-5
print("Max T", max_T)
epoch = 0
total_cost = 0

while epoch<epoch_max and not args.no_greedy: # WARNING: hard coded epochs
    epoch_loss = 0
    epoch+=1
    ugf = 0
    encoder.train()
    decoder.train()
    classifier.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device) # GPU settings

        mask_S = trainmasks[batch_idx*bs:(1+batch_idx)*bs]
        mask_new = copy.deepcopy(mask_S) # bs*1024
        mask_S = mask_S.reshape((-1,32,32)).to(device)
        mask_new[:, getT2(T)] = 1 # bs*1024
        mask_new = mask_new.reshape((-1,32,32)).to(device) # bs*32*32

        optimizer.zero_grad()
        mask_new = mask_new.reshape((-1,32,32)) # bs*32*32
        out = classifier(inputs*mask_S[:,None,:,:])
        out_new = classifier(inputs*mask_new[:,None,:,:])

        inputs_g = decoder(encoder(inputs*mask_S[:,None,:,:]))
        out_g = classifier(inputs_g*mask_new[:,None,:,:])
        ug = torch.tensor(args.ug)
        set_size = inputs.shape[1]
        # u = 1 - torch.max(out)
        # print("u",u)
        u = 0.9
        batch_loss = torch.sum(u*((ug)*loss(out_new, targets) + (1-ug)*loss(out_g, targets)) + (1-u)*loss(out, targets))
        reduced_loss = torch.sum((ug)*loss(out_new, targets) + (1-ug)*loss(out_g, targets))
        epoch_loss+=reduced_loss
        batch_loss.backward()
        optimizer.step()
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        classifier.eval()
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mask_S = trainmasks[batch_idx*bs:(1+batch_idx)*bs]
            mask_new = copy.deepcopy(mask_S) # bs*1024
            mask_new[:, getT2(T)] = 1 # bs*1024
            mask_new = mask_new.reshape((-1,32,32)).to(device) # bs*32*32

            out = classifier(inputs*mask_new[:,None,:,:])
            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(targets==Y_hat)
            total += targets.shape[0]
        acc = correct/total
        print(f"Epoch {epoch} train accuracy {acc}")
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mask_S = testmasks[batch_idx*bs:(1+batch_idx)*bs]
            mask_new = copy.deepcopy(mask_S) # bs*1024
            mask_new[:, getT2(T)] = 1 # bs*1024
            mask_new = mask_new.reshape((-1,32,32)).to(device) # bs*32*32

            out = classifier(inputs*mask_new[:,None,:,:])
            Y_hat = torch.argmax(out, dim=1)
            correct += torch.sum(targets==Y_hat)
            total += targets.shape[0]
        acc = correct/total
        print(f"Epoch {epoch} val accuracy {acc}")
    if (epoch+1)%args.greedy_freq==0 and not greedy_done:
        path = os.path.join(save_folder, f'classifier__LSH_c{c}_ns{args.num_sets}_{epoch+1}_sizeofT{len(T)}')
        path_T = os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{epoch+1}_sizeofT{len(T)}')
        with open(path_T, "wb") as f:
            pickle.dump(T, f)
        torch.save(classifier.state_dict(), path)
        start_time = time.time()
        if train_size > args.greedy_sample_size:
            sample_indices = np.random.permutation(np.arange(train_size))[:args.greedy_sample_size]
            epoch_loss = epoch_loss*args.greedy_sample_size/train_size
        else:
            sample_indices = np.random.permutation(np.arange(train_size))
        greedy_subset = torch.utils.data.Subset(data_subset, sample_indices)
        greedy_sets = [trainsets[i] for i in sample_indices]
        new_feature = greedy(encoder, decoder, classifier, loss, greedy_subset, greedy_sets, T, epoch_loss, device, args, topk=topk)
        if new_feature==None:
            if len(T)>=max_T or (epoch>epoch_max): # WARNING : hard coded max T just before greedy
                break
            continue
        end_time = time.time()
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



if args.warm_end:
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()

    if c==0:
        e = args.warm_end_load_T//5+2
        if args.warm_end_load_T==70:
            e = 15
    elif c==1:
        e = args.warm_end_load_T//5+2
    elif c==2:
        e = args.warm_end_load_T//5+2
    elif c==3:
        e = args.warm_end_load_T//2+4
        if args.warm_end_load_T==80:
            e = 43
    classifier_path = f"classifier__LSH_c{c}_ns{args.num_sets}_{e}_sizeofT{args.warm_end_load_T}"
    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{e}_sizeofT{args.warm_end_load_T}'), "rb"))

    if not args.warm_end_fresh_start:
        classifier.load_state_dict(torch.load(os.path.join(save_folder, classifier_path), map_location=device))

    if args.warm_end_loss == 'cross_entropy':
        loss = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    elif args.warm_end_loss == 'generative_loss':
        loss = nn.CrossEntropyLoss(reduction='none').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(reduction='none')
    
    # if args.warm_end_end_to_end:
    #     optimizer = optim.Adam(list(classifier.parameters())+list(encoder.parameters())+list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #     print("Training end to end")
    # else:
    #     optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #     print("Not Training end to end")
    T_val = T
    acc_last = -10

    for epoch in range(args.epochs):
        classifier.train()
        classifier.training = True
        train_loss = 0
        total = 0
        correct = 0
        optimizer = optim.SGD(classifier.parameters(), lr=learning_rate(0.1, epoch), momentum=0.9, weight_decay=5e-4)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device) # GPU settings
            mask_S = trainmasks[batch_idx*bs:(1+batch_idx)*bs]
            mask_new = copy.deepcopy(mask_S)
            mask_new[:, getT2(T_val)] = 1
            mask_new = mask_new.reshape((-1,32,32)).to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs*mask_new[:,None,:,:])               # Forward Propagation
            batch_loss = loss(outputs, targets)  # Loss
            batch_loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update

            train_loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                    %(epoch, args.epochs, batch_idx+1,
                        (len(data_subset)//bs)+1, batch_loss.item(), 100.*correct/total))
            sys.stdout.flush()
        train_acc = correct/total

        with torch.no_grad():
            classifier.eval()
            classifier.training = False
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    mask_S = testmasks[batch_idx*bs:(1+batch_idx)*bs]
                    mask_new = copy.deepcopy(mask_S)
                    # mask_new[:, getT2(T_val)] = 1
                    # mask_new = mask_new.reshape((-1,32,32)).to(device)
                    # outputs = classifier(inputs*mask_new[:,None,:,:])
                    mask_new = mask_new.repeat(3,1,1).permute(1,0,2)
                    mask_new = mask_new.reshape((-1,3072))
                    mask_new[:,T2] = 1
                    mask_new = mask_new.reshape((-1,3,32,32)).to(device)
                    outputs = classifier(inputs*mask_new)
                    batch_loss = loss(outputs, targets)

                    test_loss += batch_loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
            acc = correct/total
            if train_acc > acc_last:
                classifier_final =  copy.deepcopy(classifier)
                acc_last = train_acc
            print(f"Warm end : Epoch {epoch} val accuracy {acc}")

    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T}_warm_end_fresh_start_{args.warm_end_fresh_start}'

    classifier =  copy.deepcopy(classifier_final)    

    torch.save(classifier.state_dict(), os.path.join(save_folder, f'classifier__LSH_c{c}' + suffix))

    path_T = os.path.join(save_folder, f'T_LSH_c{c}' + suffix)
    pickle.dump(T, open(path_T, 'wb'))


if(args.test):
    if(args.warm_end_load_T >max_T):
        print(f"warm_end_load_T exceeds max possible |T| ({max_T})")
        exit()
        
    suffix = f'_ns{args.num_sets}_model_final__warm_end_end_to_end_{args.warm_end_end_to_end}__warm_end_loss_{args.warm_end_loss}__T_{args.warm_end_load_T+5}_warm_end_fresh_start_{args.warm_end_fresh_start}'
    # args.warm_end_load_T += 5
    if c==0:
        e = args.warm_end_load_T//5+2
        if args.warm_end_load_T==70:
            e = 15
    elif c==1:
        e = args.warm_end_load_T//5+2
    elif c==2:
        e = args.warm_end_load_T//5+2
    elif c==3:
        e = args.warm_end_load_T//2+4
        if args.warm_end_load_T==80:
            e = 43
    T = pickle.load(open(os.path.join(save_folder, f'T__LSH_c{c}_ns{args.num_sets}_{e}_sizeofT{args.warm_end_load_T}'), "rb"))

    classifier.load_state_dict(torch.load(os.path.join(save_folder, f'classifier__LSH_c{c}' + suffix), map_location=device))

    total = 0
    correct = 0
    Q_plus = 0
    Q_minus = 0

    T_val = T
    T_g = None
    print("val T", T_val, len(T_val))
    
    with torch.no_grad():
        classifier.eval()
        classifier.training = False
        correct = 0
        total = 0
        accuracies = [0]
        sizes = [0]
        bucket_size = int(math.ceil(len(val_indices[c])/10))
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mask_S = testmasks[batch_idx*bs:(1+batch_idx)*bs]
            mask_new = copy.deepcopy(mask_S)
            mask_new[:, getT2(T_val)] = 1
            mask_new = mask_new.reshape((-1,32,32)).to(device)
            '''
            new stuff
            '''
            # mask_new = mask_new.repeat(3,1,1).permute(1,0,2)
            # mask_new = mask_new.reshape((-1,3072))
            # mask_new[:,T2] = 1
            # mask_new = mask_new.reshape((-1,3,32,32)).to(device)
            # outputs = classifier(inputs*mask_new)
            outputs = classifier(inputs*mask_new[:,None,:,:])
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            accuracies[-1] += predicted.eq(targets.data).cpu().sum()
            sizes[-1] += targets.size(0)
            if sizes[-1] > bucket_size:
                accuracies[-1] = accuracies[-1]/sizes[-1]
                accuracies.append(0)
                sizes.append(0)
        acc = correct/total
    print(accuracies[:-1], sizes[:-1])
    with open(os.path.join(save_metrics_folder, f"result_LSH_c{c}_T{args.warm_end_load_T}.pkl"), "wb") as f:
        pickle.dump((accuracies[:-1], sizes[:-1]), f)