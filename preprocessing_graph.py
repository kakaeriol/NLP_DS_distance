## Read DAILOG DATASET
import numpy as np
import pandas as pd
import networkx as nx
from networkx import path_graph, random_layout
import os
random_state = np.random.RandomState(42)

data_path="/home/n/nguyenpk/CS6220/project/NLP_DS_distance"

def plot_graph_with_color(Graph, pos, labels, ax, colors = None, COLOR_SCHEME = 'Set1'):
    """
    """
    if colors is None:
        colors = [labels[node] for node in list(Graph.nodes())]
    nx.draw_networkx(Graph, pos=pos, labels=labels, with_labels=True, node_color=colors, cmap=COLOR_SCHEME, ax=ax)

def plot_graphs(Graph, mapping_conv, mapping_label):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
    # pos_nodes = nx.spring_layout(G)
    pos_nodes = random_layout(Graph, seed=10)
    plot_graph_with_color(Graph, pos_nodes, mapping_label, ax)
    pos_attrs = {}
    for node, coords in pos_nodes.items():
        pos_attrs[node] = (coords[0], coords[1] + -0.03)
    nx.draw_networkx_labels(Graph, pos_attrs, labels=mapping_conv,font_size = 6, ax=ax)
    # nx.draw_networkx_labels(Graph, pos_attrs, labels=mapping_conv,font_size = 8, ax=ax)
    
def create_perms(node_id, window_past=3, window_future=3):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """
    all_perms = set()
    length = len(node_id)
    array = np.arange(length)
    for j in range(length):
        perms = set()
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(length, j + window_future + 1)]
        for item in eff_array:
            perms.add((node_id[j], node_id[item]))
        all_perms = all_perms.union(perms)
    return list(all_perms)


# for iraw in ["IEMOCAP", "MELD"]:
for iraw in ["Daily_Dailog"]:
    data_link = os.path.join(data_path, "data/{}_raw.pkl".format(iraw))
    data_raw = pd.read_pickle(data_link)
    edge_type_to_idx = {}
    if iraw == "MELD":
        n_speakers = 9
    else:
        n_speakers = 2
    for j in range(n_speakers):
        for k in range(n_speakers):
            edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
            edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
    for ikey in ["train", "test", "dev"]:
        print(iraw, ikey)
        idata = data_raw[ikey]
        lenconv = 0
        window_past = 1
        window_future = 1
        map_conv_all = {}
        map_label_all = {}
        map_edge_type_all = []
        ll_G = []
        G1 = nx.DiGraph()
        speaker = [i for sublist in idata["speakers"] for i in sublist]
        for ii, data in enumerate(idata["conversation"]):
            if ii % 100 == 0:
                print(ii)
            conv = data
            emo  = idata["emotions"][ii]
            node_id = [lenconv + i for i in range(len(conv))]
            mapping_conv = dict(zip(node_id, conv))
            mapping_label = dict(zip(node_id, emo))
            perms = create_perms(node_id, window_past, window_future)
            edge_type = []
            for item in perms:
                if item[0] < item[1]:
                    c = '0'
                else:
                    c = '1'
                eid = "{}{}{}".format(speaker[item[0]], speaker[item[1]],c)
                edge_type.append(edge_type_to_idx[eid])        
            G = nx.DiGraph()
            G.add_nodes_from(node_id)
            G.add_edges_from(perms)
            ll_G.append(G)
            G1 = nx.disjoint_union(G, G1)
            lenconv = lenconv+ len(data)
            map_conv_all.update(mapping_conv)
            map_label_all.update(mapping_label)
            map_edge_type_all.append(edge_type)
        nx.write_gpickle(G1,'{}_{}.gpickle'.format(iraw, ikey))
        pickle.dump({'G':G1, 'map_conv':map_conv_all, 'map_label':map_label_all, 'map_edge_type': map_edge_type_all}, open('{}_{}_.pickle'.format(iraw, ikey), 'wb'))