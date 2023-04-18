## Read DAILOG DATASET
import numpy as np
import pandas as pd
import networkx as nx
from networkx import path_graph, random_layout
random_state = np.random.RandomState(42)

data_raw = pd.read_pickle("data/Daily_Dailog_raw.pkl")



ikey = "train"
idata = data_raw["train"]
lenconv = 0
window_past = 1
window_future = 1
map_conv_all = {}
map_label_all = {}
ll_G = []
G1 = nx.DiGraph()
for ii, data in enumerate(idata["conversation"]):
    print(ii)
    conv = data
    emo  = idata["emotions"][ii]
    node_id = [lenconv + i for i in range(len(conv))]
    mapping_conv = dict(zip(node_id, conv))
    mapping_label = dict(zip(node_id, emo))
    perms = create_perms(node_id, window_past, window_future)
    G = nx.DiGraph()
    G.add_nodes_from(node_id)
    G.add_edges_from(perms)
    ll_G.append(G)
    G1 = nx.disjoint_union(G, G1)
    lenconv = lenconv+ len(data)
    map_conv_all.update(mapping_conv)
    map_label_all.update(mapping_label)
