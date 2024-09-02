import pdb

import __init__
import os
import sys
import time
import glob
import math
import numpy as np
import torch
torch.cuda.empty_cache()
from gcn import utils
import logging
import argparse
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributions.categorical as cate
import torchvision.utils as vutils

import sampler as GraphSAINT
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from model_search import Network
from model import NetworkPPI as Model
from architect import Architect
from tensorboardX import SummaryWriter
from scipy.stats.stats import kendalltau
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


# torch_geometric.set_debug(True)
parser = argparse.ArgumentParser("ogb")
parser.add_argument('--data', type=str, default='ogb-dataset/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--batch_increase', default=1, type=int, help='how much does the batch size increase after making a decision')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=6, help='total number of cells')
parser.add_argument('--n_steps', type=int, default=3, help='total number of layers in one cell')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='ogb', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--random_seed', action='store_true', help='use seed randomly')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_dec_epoch', type=int, default=9, help='warmup decision epoch')
parser.add_argument('--decision_freq', type=int, default=7, help='decision freq epoch')
parser.add_argument('--history_size', type=int, default=4, help='number of stored epoch scores')
parser.add_argument('--use_history', action='store_true', help='use history for decision')
parser.add_argument('--in_channels', default=128, type=int, help='the channel size of input point cloud ')
parser.add_argument('--post_val', action='store_true', default=False, help='validate after each decision')
parser.add_argument('--K', default=9, help='use to search')

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.save = 'log/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = ''
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

writer = SummaryWriter(log_dir=args.save, max_queue=50)


def histogram_average(history, probs):
    histogram_inter = torch.zeros(probs.shape[0], dtype=torch.float).cuda()
    if not history:
        return histogram_inter
    for hist in history:
        histogram_inter += utils.histogram_intersection(hist, probs)
    histogram_inter /= len(history)
    return histogram_inter

def score_image(type, score, epoch):
    score_img = vutils.make_grid(
        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(score, 1), 2), 3),
        nrow=7,
        normalize=True,
        pad_value=0.5)
    writer.add_image(type + '_score', score_img, epoch)

def edge_decision(type, alphas, selected_idxs, candidate_flags, probs_history, epoch, model, args):
    mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
    importance = torch.sum(mat[:, 1:], dim=-1)
    # logging.info(type + " importance {}".format(importance))

    probs = mat[:, 1:] / importance[:, None]
    # print(type + " probs", probs)
    entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.size()[1])
    # logging.info(type + " entropy {}".format(entropy))

    if args.use_history:
        # logging.info(type + " probs history {}".format(probs_history))
        histogram_inter = histogram_average(probs_history, probs)

        probs_history.append(probs)
        if (len(probs_history) > args.history_size):
            probs_history.pop(0)

        score = utils.normalize(importance) * utils.normalize(
            1 - entropy) * utils.normalize(histogram_inter)

    else:
        score = utils.normalize(importance) * utils.normalize(1 - entropy)


    if torch.sum(candidate_flags.int()) > 0 and \
            epoch >= args.warmup_dec_epoch and \
            (epoch - args.warmup_dec_epoch) % args.decision_freq == 0:
        masked_score = torch.min(score,
                                 (2 * candidate_flags.float() - 1) * np.inf)
        selected_edge_idx = torch.argmax(masked_score)
        selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1  # add 1 since none op
        selected_idxs[selected_edge_idx] = selected_op_idx

        candidate_flags[selected_edge_idx] = False
        alphas[selected_edge_idx].requires_grad = False
        if type == 'normal':
            reduction = False
        elif type == 'reduce':
            reduction = True
        else:
            raise Exception('Unknown Cell Type')
        candidate_flags, selected_idxs = model.check_edges(candidate_flags,
                                                           selected_idxs)
        logging.info("#" * 30 + " Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}, added edge {} with op idx {}".format(epoch,
                                                                                          type,
                                                                                          selected_idxs,
                                                                                          selected_edge_idx,
                                                                                          selected_op_idx))
        print(type + "_candidate_flags {}".format(candidate_flags))
        score_image(type, score, epoch)
        return True, selected_idxs, candidate_flags

    else:
        logging.info("#" * 30 + " Not a Decision Epoch " + "#" * 30)
        logging.info("epoch {}, {}_selected_idxs {}".format(epoch,
                                                            type,
                                                            selected_idxs))
        print(type + "_candidate_flags {}".format(candidate_flags))
        score_image(type, score, epoch)
        return False, selected_idxs, candidate_flags
 
def expand_1hop_graph(graph,subgraph):
    graph= graph.to(DEVICE)
    edge_index = graph.edge_index
    subgraph_nodes = subgraph.edge_index.unique()
    print("subgraph_node",subgraph_nodes.size())
 
    num_selected_nodes = 100
    selected_nodes = subgraph_nodes[torch.randperm(subgraph_nodes.size(0))[:num_selected_nodes]].to(DEVICE)
    print("select_node",selected_nodes)
    # 选择的节点+1hop
    subgraph_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=selected_nodes,
    num_hops=1,
    edge_index=edge_index, 
    )
 
    #和子图合并
    # import pdb
    # pdb.set_trace()
    #xy信息
    subgraph_nodes=subgraph_nodes.to(DEVICE)
    subgraph.x = torch.cat([subgraph.x, graph.x[subgraph_nodes]], dim=0)
    subgraph.y = torch.cat([subgraph.y, graph.y[subgraph_nodes]], dim=0)

    #mask信息
    subgraph.train_mask =  torch.cat([subgraph.train_mask, graph.train_mask[subgraph_nodes]], dim=0)
    subgraph.valid_mask =  torch.cat([subgraph.valid_mask, graph.valid_mask[subgraph_nodes]], dim=0)
    subgraph.test_mask =  torch.cat([subgraph.test_mask, graph.test_mask[subgraph_nodes]], dim=0)
 
    # 可选：复制节点特征等其他信息
    
    # subgraph.x = subgraph_nodes
    print("subgraph_node_after",subgraph_nodes.size())
    return subgraph
     
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.random_seed:
        args.seed = np.random.randint(0, 1000, 1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)


    # dataset = PygNodePropPredDataset(name='ogbn-products', root=args.data)
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.data)
    dataset = PygNodePropPredDataset(name='ogbn-papers100M', root=args.data)

    if dataset.name == 'ogbn-products':
        args.in_channels = 100
    data = dataset[0]
    print(3)
    split = dataset.get_idx_split()
    n_classes = dataset.num_classes
    mask = torch.zeros(len(data.x), dtype=torch.bool)
    train_mask = torch.tensor(mask)
    valid_mask = torch.tensor(mask)
    test_mask = torch.tensor(mask)

    train_mask[split['train']] = True
    valid_mask[split['valid']] = True
    test_mask[split['test']] = True

    data.train_mask = train_mask
    data.valid_mask = valid_mask
    data.test_mask = test_mask

    evaluator = Evaluator('ogbn-papers100M')

    criterion = nn.CrossEntropyLoss().cuda()

    subgraphs = []
    models = []
    nets = []
    genotypes = []
    accs = []
    ranks = []
    
    # batch si
    # 采样次数 k
        
    for i in range(args.K):
        
        n = len(data.y)
        SAINT_data = GraphSAINT.GraphSAINTNodeSampler(data, batch_size=5000, num_steps=int(n/5000))
        count = 0
        subgraph = next(iter(SAINT_data))
        subgraph = subgraph.to(DEVICE)
        subgraphs.append(subgraph)
        model = Network(args.init_channels, n_classes, args.num_cells, criterion,
                        args.n_steps, in_channels=args.in_channels).cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        num_edges = model._steps * 2
        post_train = 5
        # 9 + 7*5 +5
        # args.epochs = args.warmup_dec_epoch + args.decision_freq * (num_edges - 1) + post_train + 1
        args.epochs = 10 + 15 * (num_edges - 1) + 10 + 1

        logging.info("total epochs: %d", args.epochs)

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        architect = Architect(model, args)

        normal_selected_idxs = torch.tensor(len(model.alphas_normal) * [-1], requires_grad=False, dtype=torch.int).cuda()
        normal_candidate_flags = torch.tensor(len(model.alphas_normal) * [True], requires_grad=False, dtype=torch.bool).cuda()
        # logging.info('normal_selected_idxs: {}'.format(normal_selected_idxs))
        # logging.info('normal_candidate_flags: {}'.format(normal_candidate_flags))
        model.normal_selected_idxs = normal_selected_idxs
        model.normal_candidate_flags = normal_candidate_flags

 
        count = 0
        normal_probs_history = [] 
        # for epoch in range(args.epochs):
        for epoch in range(100):
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
 
            train_acc = train(subgraph, model, architect, criterion, optimizer, lr, evaluator)
            
            test_acc, valid_acc = infer(subgraph, model, criterion, evaluator)
            logging.info('train_acc %f\tvalid_acc %f', train_acc, valid_acc)

            # make edge decisions
            saved_memory_normal, model.normal_selected_idxs, \
            model.normal_candidate_flags = edge_decision('normal',
                                                         model.alphas_normal,
                                                         model.normal_selected_idxs,
                                                         model.normal_candidate_flags,
                                                         normal_probs_history,
                                                         epoch,
                                                         model,
                                                         args)
            # logging.info("weight",model.alphas_normal)
            if saved_memory_normal:
                torch.cuda.empty_cache()
                count += 1
                new_batch_size = args.batch_size + args.batch_increase * count
                logging.info("new_batch_size = {}".format(new_batch_size))

                if args.post_val:
                    test_acc, valid_acc = infer(subgraph, model, criterion, evaluator)
                    logging.info('post valid_acc %f', valid_acc)
            # print(m odel.get_genotype())
            writer.add_scalar('stats/train_acc', train_acc, epoch)
            writer.add_scalar('stats/valid_acc', valid_acc, epoch)
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            scheduler.step()

        logging.info("#" * 30 + " Done " + "#" * 30)
        logging.info('genotype = %s', model.get_genotype())
        nets.append(model)
        genotype = model.get_genotype()
        genotypes.append(genotype)
        nets.append(model)

        model = Model(args.init_channels, n_classes, args.num_cells, False, genotype,
                       in_channels=args.in_channels).cuda()
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
 
        for i in range(200):
            model.train()
            target = subgraph.y
            n = len(target)
            optimizer_model.zero_grad()
            logits = model(subgraph)[0]
            loss = criterion(logits[subgraph.train_mask], torch.squeeze(target[subgraph.train_mask]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer_model.step() 
            y_pred = logits.argmax(dim=-1, keepdim=True)
            val_acc = evaluator.eval({
                'y_true': target[subgraph.valid_mask],
                'y_pred': y_pred[subgraph.valid_mask],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': target[subgraph.test_mask],
                'y_pred': y_pred[subgraph.test_mask],
            })['acc']
            logging.info("epoch %s val_acc %s",i,val_acc)
            logging.info("epoch %s test_acc %s",i,test_acc)
            
          
        models.append(model)
    temp = torch.zeros(size=(args.K,))
    print("subgraphs",subgraphs)
     
    for i in range(args.K):
        n = len(data.y)
        subdata = GraphSAINT.GraphSAINTNodeSampler(data, batch_size=50000, num_steps=int(n/50000))
        sum = 0
        for batch in subdata:
            batch = batch.to(DEVICE)
            acc = infer(batch, models[i], criterion, evaluator=evaluator)[0]
            num = acc*len(batch.y[batch.test_mask])
            sum += num
        acc = sum/(len(data.y[data.test_mask]))
        temp[i] = acc
    #temp: model的valid acc rank list
    
    ranks.append(temp.argsort())  
    accs.append(temp) 
     
    for i in range(args.K):
        temp = torch.zeros(size=(args.K,))
        for j in range(args.K):
            temp [j] = infer(subgraphs[i], models[j], criterion, evaluator=evaluator)[0]
        accs.append(temp)
        ranks.append(temp.argsort())

    print(accs[0])
    kens = torch.zeros(size=(args.K,))
    for i in range(args.K):
        print(genotypes[i], accs[i+1])
    
    for i in range(args.K):
        kens[i] = kendalltau(ranks[0], ranks[i+1]).correlation
    indic = kens.argmax()
    
    model = nets[indic]
    subgraph = subgraphs[indic]
     
    # expansion
    exp = 5
 
    for i in range(exp):
        model.expand() 
        subgraph=subgraph.to(DEVICE)
        subgraph=expand_1hop_graph(data,subgraph)
        subgraph=subgraph.to(DEVICE) 
        epochs = args.warmup_dec_epoch + args.decision_freq + post_train + 1
       
        for epoch in range(epochs):
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            architect = Architect(model, args)
            # training 
            train_acc = train(subgraph, model, architect, criterion, optimizer, lr, evaluator)
            test_acc, valid_acc = infer(subgraph, model, criterion, evaluator)
            logging.info('train_acc %f\tvalid_acc %f', train_acc, valid_acc)

            # make edge decisions 
            saved_memory_normal, model.normal_selected_idxs, \
            model.normal_candidate_flags = edge_decision('normal',
                                                         model.alphas_normal,
                                                         model.normal_selected_idxs,
                                                         model.normal_candidate_flags,
                                                         normal_probs_history,
                                                         epoch,
                                                         model,
                                                         args)
            if saved_memory_normal:
                torch.cuda.empty_cache()
                count += 1
                new_batch_size = args.batch_size + args.batch_increase * count
                logging.info("new_batch_size = {}".format(new_batch_size))
                if args.post_val:
                    test_acc, valid_acc = infer(subgraph, model, criterion, evaluator)
                    logging.info('post valid_acc %f', valid_acc)
            writer.add_scalar('stats/train_acc', train_acc, epoch)
            writer.add_scalar('stats/valid_acc', valid_acc, epoch)
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            scheduler.step()
        logging.info("#" * 30 + " Done " + "#" * 30)
        logging.info('genotype = %s', model.get_genotype())

def train(input, model, architect, criterion, optimizer, lr, evaluator):
    
    model.train()

    target = input.y
    n = len(target)
    target_search = input.y[input.valid_mask]
    architect.step(input, target, input, target_search, lr, optimizer, unrolled=args.unrolled, mask=True)
    optimizer.zero_grad()
    logits = model(input)
    target1 = target[input.train_mask].long()
    loss = criterion(logits[input.train_mask], torch.squeeze(target1))

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    y_pred = logits.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': target[input.train_mask],
        'y_pred': y_pred[input.train_mask],
    })['acc']
        
    # import pdb
    # pdb.set_trace()
    return train_acc

def infer(data, model, criterion, evaluator):
    model.eval()

    with torch.no_grad():
        data = data.to(DEVICE)
        out = model(data)
        if len(out) == 2:
            out = out[0]
        y_pred = out.argmax(dim=-1, keepdim=True)

        test_acc = evaluator.eval({
            'y_true': data.y[data.test_mask],
            'y_pred': y_pred[data.test_mask],
        })['acc']

        valid_acc = evaluator.eval({
            'y_true': data.y[data.valid_mask],
            'y_pred': y_pred[data.valid_mask],
        })['acc']
        n = data.y.size(0)
        loss = criterion(out[data.valid_mask], torch.squeeze(data.y[data.valid_mask]))
    return test_acc, valid_acc

if __name__ == '__main__': 
    main() 

