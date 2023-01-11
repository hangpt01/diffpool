import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler

def prepare_data(graphs, args, max_nodes=0):
        dataset_sampler = GraphSampler(graphs, normalize=False, max_num_nodes=max_nodes,
                features=args.feature_type)
        data_loader = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size=len(graphs), 
                shuffle=True,
                num_workers=args.num_workers)

        return data_loader

def prepare_train_val_graphs(graphs, args, val_idx, max_nodes=0):

        random.shuffle(graphs)
        val_size = len(graphs) // 10
        train_graphs = graphs[:val_idx * val_size]
        if val_idx < 9:
                train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
        val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]
        print('Num training graphs: ', len(train_graphs), 
                '; Num validation graphs: ', len(val_graphs))

        print('Number of graphs: ', len(graphs))
        print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
        print('Max, avg, std of graph size: ', 
                max([G.number_of_nodes() for G in graphs]), ', '
                "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
                "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))
        return train_graphs, val_graphs