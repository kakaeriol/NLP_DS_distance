import os, sys
import argparse
sys.path.append("/home/n/nguyenpk/CS6220/otdd")
import otdd
from otdd.pytorch.distance import DatasetDistance
#
import pandas as pd
import math
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import torch.nn as nn

import torch
import numpy as np
#----
seed = 100
if torch.cuda.is_available():
    generator = torch.Generator('cuda').manual_seed(seed)
    device = "cuda:0"
else:
    generator = torch.Generator().manual_seed(seed)
    device = "cpu"
    print("DEVICE ERROR")
    
from sklearn import preprocessing
class my_DataSet(Dataset):
    """
     This class create my dataset
    """
    def __init__(self, data):
        self.data = data[0]
        self.clf = preprocessing.LabelEncoder()
        self.targets = torch.tensor(self.clf.fit_transform(data[1]), dtype=int)
        self.y_raw = data[1] 
        self.len = len(self.targets)
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    def __len__(self):
        return self.len
    def __delitem__(self, key):
        self.data = np.delete(self.data, key, axis=0)
        self.targets = np.delete(self.targets, key, axis=0)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--id", type=int, required=True,
                        help="id of data")
    args = parser.parse_args()
    
    data_path = "/home/n/nguyenpk/CS6220/project/NLP_DS_distance"
    DD_DS = pd.read_pickle(os.path.join(data_path,"my_DD_ds.pickle"))
    num_array = 10
    size_of_each = np.ceil(len(DD_DS)/num_array)
    id_ =  args.id#sys.getenv('SLURM_ARRAY_TASK_ID')
    print("SLURM id: {}".format(id_))
    begin = int(size_of_each*id_)
    end  = begin + size_of_each
    end  = min(len(DD_DS), int(end))
    print("SLURM id: {}, begin: {}, end: {}".format(id_, begin, end))
    ll = []
    for i in range(begin, end, 1):
        each_i = np.zeros(len(DD_DS.keys()))
        for j in range(len(DD_DS.keys())):
            D_i = DD_DS[i]
            D_j = DD_DS[j]
            dist = DatasetDistance(D_i, D_j,
                                       inner_ot_method = 'gaussian_approx',#gaussian_approx
                                       debiased_loss = True,
                                       p = 2, entreg = 1e-1,
                                       min_labelcount=1,
                                       device='cpu') ## error device!! ## NOW USING THIS ONE TO CHECK FIRST
            try:
                d = dist.distance()
                each_i[j]  = d.item()
            except:
                continue
            print("Finish compute distance:i,j",i, j, d)
        ll.append(each_i)
    print("Begin to saving")
    print(id_)
    pd.to_pickle(ll, os.path.join(data_path, "OTDD_TRY/{}_glove.pkl".format(id_)))
