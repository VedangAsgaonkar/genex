'''
Definition of costs of all the features
'''
import pandas as pd
import numpy as np

def get_costs(dataset):
    '''
    Returns 
        a list of costs of all features
        a tuple of min cost and max cost for the initial subset
    '''
    
    # Argument 'dataset' is supposed to be either "covid" or "disease_pred"
    # The order of features is the same as the dataset pickle files, and is also
    # present in the corresponding txt files

    f = open("./data/" + dataset + ".txt", 'r+')
    cost_list = []
    min_cost, max_cost = 1e10, 0

    for line in f.readlines():
        # print(line.split(' ')[-1].replace('\n', "", 1))
        x = int(line.split(' ')[-1].replace('\n', "", 1))
        cost_list.append(x)
        max_cost = max(max_cost, x)
        min_cost = min(min_cost, x)


    return cost_list, min_cost*15, min_cost*10 

