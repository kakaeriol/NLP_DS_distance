from typing import List, Union, Any
import os
import numpy as np
import re
import pandas as pd

#### Need to change the path later
data_path = "/home/n/nguyenpk/CS6220/data/{}_token_fts.pkl"
prerain_embedd = "/home/n/nguyenpk/CS6220/data/{}_embedd.pkl"


def load_embedding(path):
    glv_pretrained = pd.read_pickle(path)
    vocab_size, e_dim = glv_pretrained['embedding'].shape
    embedding = nn.Embedding(vocab_size,e_dim)
    embedding.weight = nn.Parameter(torch.from_numpy(glv_pretrained['embedding']).float())
    return embedding
for idata in ["Daily", "MELD", "IEMOCAP"]:
    """ Saving the embedding vector GLOVE"""
    print("Loading embedding {}".format(idata))
    embedding = load_embedding(prerain_embedd.format(idata))
    data = pd.read_pickle(data_path.format(idata))

    rs = {}
    for itype in ["train", "dev", "test"]:
        print(itype)
        ll = []
        for i in  data["train"]['data_token']:
            ll.append(embedding(torch.tensor(i, dtype=int)).sum(axis=1))
        rs[itype]["data"] = ll
        rs[itype]["emotion"] = data["train"]["emotions"]
    pd.to_pickle(rs, data_path.format("Glove_embedding_{}".format(idata)))
    print("Finish, file save to {}".format(data_path.format("Glove_embedding_{}".format(idata))))