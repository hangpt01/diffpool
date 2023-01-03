import networkx as nx
import numpy as np
import scipy as sc
from itertools import combinations
import pickle as pk
import os
import re

import util

# def read_graphfile(datadir, dataname, max_nodes=None):
#     ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
#         graph index starts with 1 in file

#     Returns:
#         List of networkx objects with graph and node labels
#     '''
#     prefix = os.path.join(datadir, dataname, dataname)
#     filename_graph_indic = prefix + '_graph_indicator.txt'
#     # index of graphs that a given node belongs to
#     graph_indic={}
#     with open(filename_graph_indic) as f:
#         i=1
#         for line in f:
#             line=line.strip("\n")
#             graph_indic[i]=int(line)
#             i+=1

#     filename_nodes=prefix + '_node_labels.txt'
#     node_labels=[]
#     try:
#         with open(filename_nodes) as f:
#             for line in f:
#                 line=line.strip("\n")
#                 node_labels+=[int(line) - 1]
#         num_unique_node_labels = max(node_labels) + 1
#     except IOError:
#         print('No node labels')
 
#     filename_node_attrs=prefix + '_node_attributes.txt'
#     node_attrs=[]
#     try:
#         with open(filename_node_attrs) as f:
#             for line in f:
#                 line = line.strip("\s\n")
#                 attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
#                 node_attrs.append(np.array(attrs))
#     except IOError:
#         print('No node attributes')
       
#     label_has_zero = False
#     filename_graphs=prefix + '_graph_labels.txt'
#     graph_labels=[]

#     # assume that all graph labels appear in the dataset 
#     #(set of labels don't have to be consecutive)
#     label_vals = []
#     with open(filename_graphs) as f:
#         for line in f:
#             line=line.strip("\n")
#             val = int(line)
#             #if val == 0:
#             #    label_has_zero = True
#             if val not in label_vals:
#                 label_vals.append(val)
#             graph_labels.append(val)
#     #graph_labels = np.array(graph_labels)
#     label_map_to_int = {val: i for i, val in enumerate(label_vals)}
#     graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
#     #if label_has_zero:
#     #    graph_labels += 1
    
#     filename_adj=prefix + '_A.txt'
#     adj_list={i:[] for i in range(1,len(graph_labels)+1)}    
#     index_graph={i:[] for i in range(1,len(graph_labels)+1)}
#     num_edges = 0
#     with open(filename_adj) as f:
#         for line in f:
#             line=line.strip("\n").split(",")
#             e0,e1=(int(line[0].strip(" ")),int(line[1].strip(" ")))
#             adj_list[graph_indic[e0]].append((e0,e1))
#             index_graph[graph_indic[e0]]+=[e0,e1]
#             num_edges += 1
#     for k in index_graph.keys():
#         index_graph[k]=[u-1 for u in set(index_graph[k])]

#     graphs=[]
#     for i in range(1,1+len(adj_list)):
#         # indexed from 1 here
#         G=nx.from_edgelist(adj_list[i])
#         if max_nodes is not None and G.number_of_nodes() > max_nodes:
#             continue
      
#         # add features and labels
#         G.graph['label'] = graph_labels[i-1]
#         for u in util.node_iter(G):
#             if len(node_labels) > 0:
#                 node_label_one_hot = [0] * num_unique_node_labels
#                 node_label = node_labels[u-1]
#                 node_label_one_hot[node_label] = 1
#                 util.node_dict(G)[u]['label'] = node_label_one_hot
#             if len(node_attrs) > 0:
#                 util.node_dict(G)[u]['feat'] = node_attrs[u-1]
#         if len(node_attrs) > 0:
#             G.graph['feat_dim'] = node_attrs[0].shape[0]

#         # relabeling
#         mapping={}
#         it=0
#         for n in util.node_iter(G):
#             mapping[n]=it
#             it+=1
            
#         # indexed from 0
#         graphs.append(nx.relabel_nodes(G, mapping))
#     return graphs

def read_bio_file (path_data_paper='data/atac-gex/paperdata/', path_gen_locus='data/atac-gex/raw/'):
    atac_train = sc.read_h5ad(path_data_paper + 'atac_train.h5ad')
    atac_test = sc.read_h5ad(path_data_paper + 'atac_test.h5ad')
    gex_train = sc.read_h5ad(path_data_paper + 'gex_train.h5ad')
    gex_test = sc.read_h5ad(path_data_paper + 'gex_test.h5ad')

def get_adj_list(gene_locus_):
    adj_list = []
    for k, v in gene_locus_.items():
        if len(v) > 0:
            combs = list(combinations(range(len(v)),2))
            for i,j in combs:
                adj_list.append((v[i],v[j]))
        # if k>5:
        #     break
    return adj_list

def get_graphs():
    '''
        graph_labels: GEX vector
    '''
    atac_train_path = 'data/atac-gex/paperdata/atac_train.h5ad'
    atac_test_path = 'data/atac-gex/paperdata/atac_test.h5ad'
    gex_train_path = 'data/atac-gex/paperdata/gex_train.h5ad'
    gex_test_path = 'data/atac-gex/paperdata/gex_test.h5ad'

    atac_train = sc.read_h5ad(atac_train_path)
    atac_test = sc.read_h5ad(atac_test_path)
    gex_train = sc.read_h5ad(gex_train_path)
    gex_test = sc.read_h5ad(gex_test_path)
    
    gene_locus = pk.load(open('raw/gene locus 2.pkl', 'rb'))
    gene_locus_int = pk.load(open('raw/gene locus int.pkl', 'rb'))    
    
    atac_train_np = atac_train.X.toarray()
    atac_test_np = atac_test.X.toarray()
    gex_train_np = gex_train.X.toarray()
    gex_test_np = gex_test.X.toarray() 
    
    graphs = []
    # for i in range(atac_test_np.shape[0]*39): # train+val: test_ratio=0.025*total
    # test with 1/100: 10/1000 gex vector first
    for i in range(10*39): # train+val: test_ratio=0.025*total    
        node_list = list(range(atac_train_np.shape[1]))
        G=nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(get_adj_list(gene_locus_int))
        for u in G.nodes:
            G.nodes[u]['feat'] = atac_train_np[i,:][u]
        
        G.graph['feat_dim'] = 1
        G.graph['label'] = gex_train_np[i,:]
        graphs.append(G)
        
    # for i in range(atac_test_np.shape[0]): # test
    for i in range(10): # test
        node_list = list(range(atac_train_np.shape[1]))
        G=nx.Graph()
        G.add_nodes_from(node_list)
        G.add_edges_from(get_adj_list(gene_locus_int))
        for u in G.nodes:
            G.nodes[u]['feat'] = atac_test_np[i,:][u]
        
        G.graph['feat_dim'] = 1
        G.graph['label'] = gex_test_np[i,:]
        graphs.append(G)
    
    return graphs