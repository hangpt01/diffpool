import networkx as nx
import numpy as np
import random

import gen.feat as featgen
import util

def gen_ba(n_range, m_range, num_graphs, feature_generator=None):
    graphs = []
    for i in np.random.choice(n_range, num_graphs):
        for j in np.random.choice(m_range, 1):
            graphs.append(nx.barabasi_albert_graph(i,j))

    if feature_generator is None:
        feature_generator = ConstFeatureGen(0)
    for G in graphs:
        feature_generator.gen_node_features(G)
    return graphs

def gen_2community_ba(n_range, m_range, num_graphs, inter_prob, feature_generators):
    ''' Each community is a BA graph.
    Args:
        inter_prob: probability of one node connecting to any node in the other community.
    '''

    if feature_generators is None:
        mu0 = np.zeros(10)
        mu1 = np.ones(10)
        sigma0 = np.ones(10, 10) * 0.1
        sigma1 = np.ones(10, 10) * 0.1
        fg0 = GaussianFeatureGen(mu0, sigma0)
        fg1 = GaussianFeatureGen(mu1, sigma1)
    else:
        fg0 = feature_generators[0]
        fg1 = feature_generators[1] if len(feature_generators) > 1 else feature_generators[0]

    graphs1 = []
    graphs2 = []
    #for (i1, i2) in zip(np.random.choice(n_range, num_graphs), 
    #                    np.random.choice(n_range, num_graphs)):
    #    for (j1, j2) in zip(np.random.choice(m_range, num_graphs), 
    #                        np.random.choice(m_range, num_graphs)):
    graphs0 = gen_ba(n_range, m_range, num_graphs, fg0)
    graphs1 = gen_ba(n_range, m_range, num_graphs, fg1)
    graphs = []
    for i in range(num_graphs):
        G = nx.disjoint_union(graphs0[i], graphs1[i])
        n0 = graphs0[i].number_of_nodes()
        for j in range(n0):
            if np.random.rand() < inter_prob:
                target = np.random.choice(G.number_of_nodes() - n0) + n0
                G.add_edge(j, target)
        graphs.append(G)
    return graphs
