import torch
import random
import numpy as np
from collections import defaultdict

def fetch_gaussian_hyperplanes(hplane_shape):
    hplanes = np.random.normal(size=hplane_shape)
    return hplanes

class LSH():
    def __init__(self, hash_code_dim, subset_size, embed_dim, num_hash_tables):
        self.hash_code_dim = hash_code_dim
        self.subset_size = subset_size
        self.embed_dim = embed_dim
        self.gauss_hplanes_cos = fetch_gaussian_hyperplanes((embed_dim, hash_code_dim))
        self.num_hash_tables = num_hash_tables
        self.powers_of_two = torch.from_numpy(1 << np.arange(self.subset_size - 1, -1, -1)).type(torch.FloatTensor)
        torch.random.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        self.init_hash_functions()

    def init_hash_functions(self):
        self.hash_functions = torch.LongTensor([])
        hash_code_dim = self.hash_code_dim
        indices = list(range(hash_code_dim))
        for i in range(self.num_hash_tables):
            random.shuffle(indices)
            self.hash_functions= torch.cat((self.hash_functions,torch.LongTensor([indices[:self.subset_size]])),dim=0)

    def preprocess_hashcodes(self,all_hashcodes): 
        all_hashcodes = torch.sign(all_hashcodes)
        if (torch.sign(all_hashcodes)==0).any(): 
            all_hashcodes[all_hashcodes==0]=1
        return all_hashcodes

    def index_corpus(self, corpus_embeds):
        self.corpus_embeds = corpus_embeds
        self.corpus_hashcodes = self.fetch_RH_hashcodes(self.corpus_embeds)
        self.num_corpus_items = self.corpus_embeds.shape[0]
        self.hashcode_mat = self.preprocess_hashcodes(self.corpus_hashcodes)
        self.bucketify()

    def fetch_RH_hashcodes(self, embeds):
        hcode_list = []
        projections = embeds@self.gauss_hplanes_cos
        hcode_list.append(torch.tensor(np.sign(projections)))
        hashcodes = torch.cat(hcode_list)
        return hashcodes

    def assign_bucket(self,function_id,node_hash_code):
        func = self.hash_functions[function_id]
        
        # convert sequence of -1 and 1 to binary by replacing -1 s to 0
        binary_id = torch.max(torch.index_select(node_hash_code,dim=0,index=func),torch.LongTensor([0]))
        #map binary sequence to int which is bucket Id
        bucket_id = self.powers_of_two@binary_id.type(torch.FloatTensor).to(self.powers_of_two)
        return bucket_id.item()

    def bucketify(self):
        self.all_hash_tables = []
        for func_id in range(self.num_hash_tables): 
            hash_table = defaultdict(list)#{}
            for item in range(self.num_corpus_items):
                hash_table[self.assign_bucket(func_id,self.hashcode_mat[item])].append(item)
            self.all_hash_tables.append(hash_table)

    def retrieve(self, q_embed):
        candidate_list = []
        q_hashcode =  self.preprocess_hashcodes(self.fetch_RH_hashcodes(q_embed))
        for table_id in range(self.num_hash_tables): 
            #identify bucket 
            bucket_id = self.assign_bucket(table_id,q_hashcode)
            return bucket_id
            candidate_list.extend(self.all_hash_tables[table_id][bucket_id])
        return candidate_list