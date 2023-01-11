import networkx as nx
import numpy as np
import scanpy as sc
from itertools import combinations
import pickle as pk
import random
import copy
# import os
# import re

# import util

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

# def get_atac_gex_indices(gene_locus_, gene_locus_int_, group_id=0):
#     group_genes = pk.load(open('data/atac-gex/raw/group_genes.pkl', 'rb'))
#     # pw = pk.load(open('data/atac-gex/raw/pw.pkl', 'rb'))
#     ls_atac_indices = []
#     ls_gex_indices = []
#     atac_gex_dict = {}
#     list_genes = list(gene_locus_.keys())
#     for gene in group_genes[group_id]:
#         if gene in list_genes:
#             gene_id = list_genes.index(gene)
#             atac_gex_dict[gene_id] = gene_locus_int_[gene_id]
#             for i in gene_locus_int_[gene_id]:
#                 ls_atac_indices.append(i)
#             ls_gex_indices.append(gene_id)
#     return atac_gex_dict, ls_atac_indices, ls_gex_indices        

def get_adj_list(gene_locus_):
    adj_list = []
    list_1_atac = []
    for k, v in gene_locus_.items():
        if len(v) > 0:
            if len(v) == 1:
                list_1_atac.append(v[0])
            else:    
                combs = list(combinations(range(len(v)),2))
                for i,j in combs:
                    adj_list.append((v[i],v[j]))

    return adj_list, list_1_atac

def create_inter_cluster_edges(inter_gene_list_, gene_locus_int_):
    ls_edge = []
    for (gene_id1, gene_id2) in inter_gene_list_:
        atac1 = gene_locus_int_[gene_id1]
        atac2 = gene_locus_int_[gene_id2]
        if len(atac1) > 0 and len(atac2) > 0:
            i = random.choice(atac1)
            j = random.choice(atac2)
            ls_edge.append((i,j))
        
    return ls_edge

def get_graphs(num_test_graphs=40, group_id=0):
    import time
    '''
        graph_labels: GEX vector
    '''
    start_time = time.time()
    atac_train_path = 'data/atac-gex/paperdata/chr1/atac_train_chr1.h5ad'
    atac_test_path = 'data/atac-gex/paperdata/chr1/atac_test_chr1.h5ad'
    gex_train_path = 'data/atac-gex/paperdata/chr1/gex_train_chr1.h5ad'
    gex_test_path = 'data/atac-gex/paperdata/chr1/gex_test_chr1.h5ad'

    atac_train = sc.read_h5ad(atac_train_path)
    atac_test = sc.read_h5ad(atac_test_path)
    gex_train = sc.read_h5ad(gex_train_path)
    gex_test = sc.read_h5ad(gex_test_path)
    
    # gene_locus = pk.load(open('data/atac-gex/raw/gene_locus.pkl', 'rb'))
    gene_locus_int = pk.load(open('data/atac-gex/paperdata/chr1/gene_locus_int.pkl', 'rb'))
    
    path_way = pk.load(open('data/atac-gex/paperdata/chr1/pw.pkl', 'rb'))   
    
    atac_train_np = atac_train.X.toarray()
    atac_test_np = atac_test.X.toarray()
    gex_train_np = gex_train.X.toarray()
    gex_test_np = gex_test.X.toarray() 
    
    graphs = []
    # for i in range(atac_test_np.shape[0]*39): # train+val: test_ratio=0.025*total
    # test with 1/200: 5/1000 gex vector first
    # atac_gex_dict, ls_atac_indices, ls_gex_indices = get_atac_gex_indices(gene_locus, gene_locus_int, group_id)   
    
    pw_np = np.array(path_way)
    source = pw_np[0,:]
    des = pw_np[1,:]
    inter_gene_edges = list(zip(source, des))
    
    # node_list = list(range(len(ls_atac_indices)))
    adj_list, list_1_atac = get_adj_list(gene_locus_int)
    inter_cluster_edges_list =  create_inter_cluster_edges(inter_gene_edges, gene_locus_int)
    # print(len(node_list),len(adj_list),len(inter_cluster_edges_list))
    print(len(list(gene_locus_int)), len(adj_list), len(inter_cluster_edges_list))
    
    # base_G = nx.Graph()
    # base_G.add_nodes_from(node_list)
    # base_G.add_edges_from(adj_list)
    # base_G.add_edges_from(inter_cluster_edges_list)
    print(time.time() - start_time)
    start_time = time.time()
    for i in range(num_test_graphs): # train+val: test_ratio=0.025*total    
        # G = copy.deepcopy(base_G)
        
        G = nx.Graph()
        G.add_nodes_from(list_1_atac)
        G.add_edges_from(adj_list)
        # print(G.number_of_nodes())
        G.add_edges_from(inter_cluster_edges_list)
        # print(G.number_of_nodes())
        
        # print(G.number_of_nodes())
        for u in G.nodes:
            G.nodes[u]['feat'] = atac_train_np[i,u]
            # print(G.nodes[u]['feat'])
        
        G.graph['feat_dim'] = 1
        G.graph['label'] = gex_train_np[i,:]
        # print(G.graph['label'].shape)   # num_genes
        
        # relabeling
        mapping={}
        it=0
        for n in G.nodes:
            mapping[n]=it
            it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    
    start_time = time.time()
    # for i in range(atac_test_np.shape[0]): # test
    for i in range(num_test_graphs): # test
        # G = copy.deepcopy(base_G)
        
        G = nx.Graph()
        G.add_nodes_from(list_1_atac)
        # print(atac_train_np.shape[1])
        G.add_edges_from(adj_list)
        # print(G.number_of_nodes())
        G.add_edges_from(inter_cluster_edges_list)
        # print(G.number_of_nodes())
        
        # print(G.number_of_nodes())
        for u in G.nodes:
            G.nodes[u]['feat'] = atac_test_np[i,u]
            # print(G.nodes[u]['feat'])
        
        G.graph['feat_dim'] = 1
        G.graph['label'] = gex_test_np[i,:]
        # print(G.graph['label'].shape)   # num_genes
        
        # relabeling
        mapping={}
        it=0
        for n in G.nodes:
            mapping[n]=it
            it+=1
            
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
    print(time.time() - start_time)
    print("Done loading graphs")
    return graphs, gene_locus_int      

# from cross_val import prepare_val_data

if __name__ == '__main__':
    get_graphs(num_test_graphs=1)
