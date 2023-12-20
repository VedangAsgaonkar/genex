import numpy as np
import math
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def unique_elements(list_):
    return list(set(list_))

def get_samples_with_features(feature_ids, dataset):
    '''
    inputs:
        feature indices, pytorch dataset, torch device
    returns:
        list of all samples with the specified feature values (other elements will be zero)
    '''
    num_samples = len(dataset)

    dim = dataset[0][0].shape[0]    # assuming dataset[0] = (x,y)
    z_ids = [i for i in range(dim) if i not in feature_ids]
    # samples = np.zeros((num_samples, dim))
    samples = []
    labels = []
    masked_samples = []
    # dl = Dataloader(dataset, batch_size=512, shuffle=False)
    # for i, batch in enumerate(dl):

    for i in range(len(dataset)):
        x, y = dataset[i]

        # temp = np.zeros(x.shape[0])
        # for i in range(x.shape[0]):
        #     temp[i] = x[i]
        # temp[z_ids] = 0
        temp = copy.deepcopy(x)
        temp[z_ids] = 0
        samples.append(x)
        labels.append(y)
        masked_samples.append(temp)

    return torch.Tensor(masked_samples), torch.Tensor(samples), torch.Tensor(labels)


def get_samples_with_features_batch(feature_ids, indices, dataset):
    samples = []
    dim = dataset[0][0].shape[0]
    for i in range(len(feature_ids)):
        id = indices[i]
        s = feature_ids[i]
        x = dataset[id][0]
        temp = copy.deepcopy(x)
        z_ids = [i for i in range(dim) if i not in s]
        temp[z_ids] = 0
        samples.append(temp)
    
    return samples

def set_to_mask(s, max_size):
    # s is a set
    mask = torch.zeros(max_size)
    mask[s] = 1
    return mask


def calculate_loss(transformer_encoder, set_embedding, classifier, decoder, VAE_sample, X, mask, mask_new, args, data_std_dev, eval=False, testing = False):
    if(not testing):
        encoded, _ = transformer_encoder(set_embedding(X, mask_new, free_mem= eval, eval=eval), free_mem=eval)
        # print("1",torch.cuda.memory_allocated(4))
        out_new = classifier(encoded)
        # print("2",torch.cuda.memory_allocated(4))
        z, _, _ = VAE_sample(transformer_encoder, set_embedding(X,  mask,free_mem=eval, eval=eval), args.latent_dim, eval=eval )
        # print("3",torch.cuda.memory_allocated(4))
        X_g = decoder(z) 
    else:
        X_g = X
    
    # print("4",torch.cuda.memory_allocated(4))
    encoded, _ = transformer_encoder(set_embedding(X_g, mask_new,free_mem= True, eval=eval), free_mem=eval)
    out_g = classifier(encoded)
    gen_std_dev = torch.std(X_g, axis=0)
    # print(torch.cuda.memory_allocated(4))
    if args.fix_ug:
        ug = torch.tensor(args.ug)
        set_size = X.shape[1]
        vec_ug = torch.ones(set_size)*ug
    else:
        vec_ug = gen_std_dev**2/data_std_dev**2
        ug = torch.mean(vec_ug)
        ug = torch.clamp(ug, 0,1)
    return ug, vec_ug, out_new, out_g

def greedy(transformer_encoder, decoder, classifier, set_embedding, VAE_sample, loss, data, labels, sets, T, epoch_loss, device, args, testing = False):
    batch_size = len(data)
    set_size = data[0].shape[0]
    mask = torch.stack([set_to_mask(s, set_size) for s in sets]) # shape [batch_size, set_size]
    mask_T = set_to_mask(T, set_size)
    # print(torch.sum(mask, dim=0))
    complement_mask = 1-torch.gt(mask.sum(dim=0)+mask_T,0).int()
    if batch_size > args.greedy_sample_size:
        sample_indices = np.random.permutation(np.arange(batch_size))[:args.greedy_sample_size]
        epoch_loss = epoch_loss*args.greedy_sample_size/batch_size
        data = data[sample_indices]
        labels = labels[sample_indices]
        sets = [sets[i] for i in sample_indices]
        batch_size = args.greedy_sample_size
    set_size = data[0].shape[0]
    new_feature = torch.eye(set_size).repeat(batch_size,1,1)
    mask = torch.stack([set_to_mask(s, set_size) for s in sets]) # shape [batch_size, set_size]
    mask[:,T] = 1
    mask = mask.repeat(set_size,1,1) # [set_size, bs, set_size]
    mask = mask.transpose(0,1) # mask is repeated along dim=1
    # dim 1 is for new element added
    new_mask = torch.logical_or(mask, new_feature).int() # [bs, set_size, set_size]
    filter_mask = torch.gt(torch.sum(new_mask-mask, dim=2),0) # [bs, set_size] True, False
    filter_mask = filter_mask.reshape(-1)
    X = data # [bs, set_size]
    X = torch.from_numpy(X)
    X = X/args.norm
    X = X.to(device)
    data_std_dev = torch.std(X, dim=0)
    new_X = X[:,None,:]*new_mask # [bs, set_size, set_size]
    
    new_X = new_X.reshape((-1, set_size))  
    new_mask = new_mask.reshape((-1, set_size)) 
    mask = mask.reshape((-1, set_size))
    labels = labels.repeat(set_size).reshape(batch_size*set_size)
    labels = torch.from_numpy(labels).to(device)
    
    labels = labels[filter_mask]
    new_X = new_X[filter_mask] 
    new_mask = new_mask[filter_mask]
    mask = mask[filter_mask]
    # print(new_X.shape, labels.shape, new_mask.shape, mask.shape)
    # output [bs, set_size]    
    losses = torch.Tensor()
    transformer_encoder.eval()
    set_embedding.eval()
    decoder.eval()
    classifier.eval()
    bs = args.batch_size*300 
    num_batches = math.ceil(new_X.shape[0]/bs)
    with torch.no_grad():
        for i in range(num_batches):
            X = new_X[i*bs:(i+1)*bs]
            Y = labels[i*bs:(i+1)*bs].long()
            mask_new = new_mask[i*bs:(i+1)*bs]
            mask_S = mask[i*bs:(i+1)*bs]
            ug, ugvec, out_new, out_g = calculate_loss(transformer_encoder, set_embedding, classifier, decoder, VAE_sample, X, mask_S, mask_new, args, data_std_dev, eval=True, testing = testing)
            batch_loss = (ug)*loss(out_new, Y) + (1-ug)*loss(out_g, Y)
            losses = torch.cat([losses, batch_loss])
            del ug, ugvec, out_new, out_g
            print(f"Batch {i}/{num_batches}")
    final_losses = torch.zeros(batch_size*set_size)
    # filter_mask = filter_mask
    final_losses[filter_mask] = losses
    final_losses[~filter_mask] = epoch_loss/batch_size
    final_losses = final_losses.reshape((batch_size, set_size))
    # print(final_losses.shape)
    final_losses = final_losses.sum(dim=0)
    # print(epoch_loss)
    # print(final_losses, final_losses.shape)
    gain = (epoch_loss - final_losses)*complement_mask - (1-complement_mask)*epoch_loss*args.greedy_threshold
    # print(gain, gain.shape)
    # if gain.max() > -epoch_loss*args.greedy_threshold :
    #     chosen_feature = gain.argmax().item()
    #     print("Chosen feature", chosen_feature)
    #     return chosen_feature
    # else:
    #     return None
    chosen_feature = gain.argmax().item()
    print("Chosen feature", chosen_feature)
    return chosen_feature

def greedy_test(transformer_encoder, decoder, classifier, set_embedding, data, sets, T, epoch_loss, device, args):
    """data here is generated data for values other than those is S"""
    batch_size = len(data)
    set_size = data[0].shape[0]
    mask = torch.stack([set_to_mask(s, set_size) for s in sets]) # shape [batch_size, set_size]
    mask_T = set_to_mask(T, set_size)
    # print(torch.sum(mask, dim=0))
    complement_mask = 1-torch.gt(mask.sum(dim=0)+mask_T,0).int()
    if batch_size > args.greedy_sample_size:
        sample_indices = np.random.permutation(np.arange(batch_size))[:args.greedy_sample_size]
        data = data[sample_indices]
        epoch_loss = epoch_loss*args.greedy_sample_size/batch_size
        sets = [sets[i] for i in sample_indices]
        batch_size = args.greedy_sample_size
    set_size = data[0].shape[0]
    new_feature = torch.eye(set_size).repeat(batch_size,1,1)
    mask = torch.stack([set_to_mask(s, set_size) for s in sets]) # shape [batch_size, set_size]
    mask[:,T] = 1
    mask = mask.repeat(set_size,1,1) # [set_size, bs, set_size]
    mask = mask.transpose(0,1) # mask is repeated along dim=1
    # dim 1 is for new element added
    new_mask = torch.logical_or(mask, new_feature).int() # [bs, set_size, set_size]
    filter_mask = torch.gt(torch.sum(new_mask-mask, dim=2),0) # [bs, set_size] True, False
    filter_mask = filter_mask.reshape(-1)
    X = data # [bs, set_size]
    X = X.to(device)
    data_std_dev = torch.std(X, dim=0)
    new_X = X[:,None,:]*new_mask # [bs, set_size, set_size]
    
    new_X = new_X.reshape((-1, set_size))  
    new_mask = new_mask.reshape((-1, set_size)) 
    mask = mask.reshape((-1, set_size))
    
    new_X = new_X[filter_mask] 
    new_mask = new_mask[filter_mask]
    # print(new_X.shape, labels.shape, new_mask.shape, mask.shape)
    # output [bs, set_size]    
    losses = torch.Tensor()
    transformer_encoder.eval()
    set_embedding.eval()
    decoder.eval()
    classifier.eval()
    bs = args.batch_size*50 
    num_batches = math.ceil(new_X.shape[0]/bs)
    with torch.no_grad():
        for i in range(num_batches):
            X = new_X[i*bs:(i+1)*bs]
            mask_new = new_mask[i*bs:(i+1)*bs]
            encoded, _ = transformer_encoder(set_embedding(X, mask_new))
            out = classifier(encoded)
            batch_loss = -torch.log(torch.max(out, dim=1)[0])
            losses = torch.cat([losses, batch_loss])
            # print(f"Batch {i}/{num_batches}")
    final_losses = torch.zeros(batch_size*set_size)
    # filter_mask = filter_mask
    final_losses[filter_mask] = losses
    final_losses[~filter_mask] = epoch_loss/batch_size
    final_losses = final_losses.reshape((batch_size, set_size))
    # print(final_losses.shape)
    final_losses = final_losses.sum(dim=0)
    gain = (epoch_loss - final_losses)*complement_mask - (1-complement_mask)*epoch_loss*args.greedy_threshold
    if gain.max() > 0 :
        chosen_feature = gain.argmax().item()
        return chosen_feature
    else:
        return None
