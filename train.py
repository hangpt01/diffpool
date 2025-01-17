import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

from tqdm import tqdm

import argparse
import os
import pickle
import random
import shutil
import time
import sys

import cross_val
import encoders
import gen.feat as featgen
import gen.data as datagen
from graph_sampler import GraphSampler
import load_data
import load_data_
import util

import warnings
warnings.filterwarnings("ignore")
def divide_graphs(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def evaluate(val_graphs, max_num_nodes, model, args, name='Validation', max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    with torch.no_grad():
        # random.shuffle(val_graphs)
        graphs = list(divide_graphs(val_graphs, args.batch_size))
        for graph in graphs:
            dataset = cross_val.prepare_data(graph, args, max_num_nodes)
            for batch_idx, data in enumerate(dataset):
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float()).cuda()
                labels.append(data['label'].long().numpy())
                batch_num_nodes = data['num_nodes'].int().numpy()
                assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

                ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
                preds.append(ypred.cpu().data.numpy())
                # _, indices = torch.max(ypred, 1)
                # preds.append(indices.cpu().data.numpy())

                # if max_num_examples is not None:
                #     if (batch_idx+1)*args.batch_size > max_num_examples:
                #         break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    # result = {'prec': metrics.precision_score(labels, preds, average='macro'),
    #           'recall': metrics.recall_score(labels, preds, average='macro'),
    #           'acc': metrics.accuracy_score(labels, preds),
    #           'F1': metrics.f1_score(labels, preds, average="micro")}
    # print(name, " accuracy:", result['acc'])
    result = metrics.mean_squared_error(labels, preds)
    print(name, "MSE loss: ", result)
    return result

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        # name = args.dataset
        name = 'ATAC-GEX'
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name

def gen_train_plt_name(args):
    return 'results/' + gen_prefix(args) + '.png'

def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i+1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)


def train(train_graphs, max_num_nodes, model, args, same_feat=True, val_graphs=None, test_graphs=None, writer=None,
        mask_nodes = True):
    writer_batch_idx = [0, 3, 6, 9]
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []
    for epoch in tqdm(range(args.num_epochs), desc="training"):
        total_time = 0
        avg_loss = 0.0
        model.train()
        random.shuffle(train_graphs)
        graphs = list(divide_graphs(train_graphs, args.batch_size))
        for graph in graphs:
            dataset = cross_val.prepare_data(graph, args, max_num_nodes)
            for batch_idx, data in enumerate(dataset):
                begin_time = time.time()
                model.zero_grad()
                adj = Variable(data['adj'].float(), requires_grad=False)
                # print("Size",sys.getsizeof(adj))    
                adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
                label = Variable(data['label'].long()).cuda()
                batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
                assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

                ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
                if not args.method == 'soft-assign' or not args.linkpred:
                    loss = model.loss(ypred, label)
                else:
                    loss = model.loss(ypred, label, adj, batch_num_nodes)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                iter += 1
                # avg_loss += loss
                avg_loss += loss * len(data)
                #if iter % 20 == 0:
                #    print('Iter: ', iter, ', loss: ', loss.data[0])
                elapsed = time.time() - begin_time
                total_time += elapsed

            # log once per XX epochs
            # if epoch % 10 == 0 and batch_idx == len(dataset) // 2 and args.method == 'soft-assign' and writer is not None:
            #     log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
        #         if args.log_graph:
        #             log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        # avg_loss /= batch_idx + 1
        avg_loss = avg_loss / len(dataset)
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss.item(), '; epoch time: ', total_time)
        result = evaluate(train_graphs, max_num_nodes, model, args, name='Train', max_num_examples=100)
        # train_accs.append(result['acc'])
        train_accs.append(result)
        train_epochs.append(epoch)
        if val_graphs is not None:
            val_result = evaluate(val_graphs, max_num_nodes, model, args, name='Validation')
            val_accs.append(val_result)
        if val_result > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss.item()
        if test_graphs is not None:
            test_result_ = evaluate(test_graphs, max_num_nodes, model, args, name='Test')
            test_result['acc'] = test_result_
            test_result['epoch'] = epoch
        # if writer is not None:
        #     writer.add_scalar('acc/train_acc', result['acc'], epoch)
        #     writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
        #     writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
        #     if test_dataset is not None:
        #         writer.add_scalar('acc/test_acc', test_result['acc'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
        if test_graphs is not None:
            print('Test result: ', test_result)
            test_epochs.append(test_result['epoch'])
            test_accs.append(test_result['acc'])


    return model, val_accs


def benchmark_task_val(args, writer=None, feat='node-feat'):
    all_vals = []
    if os.path.exists('graphs.pkl'):
        with open('graphs.pkl', 'rb') as handle:
            data = pickle.load(handle)
        graphs = data['graphs']
        gene_locus_int = data['gene']
    else:
        graphs, gene_locus_int = load_data_.get_graphs(num_test_graphs=20)
        with open('graphs.pkl', 'wb') as handle:
            pickle.dump({'graphs': graphs, 'gene': gene_locus_int}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#     graphs = load_data.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    example_node = util.node_dict(graphs[0])[0]
    
    # if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
    if feat == 'node-feat':
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)

    # for i in range(10):     #10-fold cross-validation
    for i in range(1):
        # train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
        #         cross_val.prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        max_num_nodes = args.max_nodes
        input_dim = graphs[0].graph['feat_dim']
        assign_input_dim = input_dim
        if max_num_nodes == 0:
            max_num_nodes = max([G.number_of_nodes() for G in graphs])
        # print(len(train_dataset), len(val_dataset))
        if args.method == 'soft-assign':
            print('Method: soft-assign')
            model = encoders.SoftPoolingGcnEncoder(
                    max_num_nodes, 
                    input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                    args.hidden_dim, gene_locus_int, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                    bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                    assign_input_dim=assign_input_dim).cuda()
        train_graphs, val_graphs = cross_val.prepare_train_val_graphs(graphs, args, i, args.max_nodes)
        _, val_accs = train(train_graphs, max_num_nodes, model, args, val_graphs=val_graphs)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.min(all_vals))
    print(np.argmin(all_vals))
    
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname', 
            help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')


    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
            const=False, default=True,
            help='Whether disable log graph')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(
                        # datadir='data',
                        logdir='log',
                        # dataset='syn1v2',
                        # max_nodes=60263,
                        max_nodes=5725,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=1,
                        num_epochs=200,
                        train_ratio=0.85,
                        test_ratio=0.025,
                        num_workers=0,
                        input_dim=512,
                        hidden_dim=128,
                        output_dim=512,
                        # num_classes=2,
                        # num_classes=12348,
                        num_classes=1235,
                        num_gc_layers=2,
                        dropout=0.0,
                        method='soft-assign',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    #writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)

    benchmark_task_val(prog_args, writer=writer)

#     if prog_args.bmname is not None:
#         benchmark_task_val(prog_args, writer=writer)
    # elif prog_args.pkl_fname is not None:
    #     pkl_task(prog_args)
    # elif prog_args.dataset is not None:
    #     if prog_args.dataset == 'syn1v2':
    #         syn_community1v2(prog_args, writer=writer)
    #     if prog_args.dataset == 'syn2hier':
    #         syn_community2hier(prog_args, writer=writer)

    writer.close()

if __name__ == "__main__":
    main()

