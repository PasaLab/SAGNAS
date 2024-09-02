import __init__
import os
import sys
import time
import glob
import math
import numpy as np
import torch
torch.cuda.empty_cache()
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
from gcn import utils
import sampler as GraphSAINT
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from model_search1 import Network
from model import NetworkPPI as Model
from architect_author import Architect
from tensorboardX import SummaryWriter
from scipy.stats.stats import kendalltau
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Coauthor
import torch_geometric.transforms as T
from deepsnap.dataset import GraphDataset
from torch.autograd import Variable

# torch_geometric.set_debug(True)
parser = argparse.ArgumentParser("ogb")
parser.add_argument('--data', type=str, default='coauthor', help='dataset name')
parser.add_argument('--root', type=str, default='./', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--batch_increase', default=1, type=int, help='how much does the batch size increase after making a decision')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=128, help='num of init channels')
parser.add_argument('--num_cells', type=int, default=4, help='total number of cells')
parser.add_argument('--n_steps', type=int, default=3, help='total number of layers in one cell')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='coauthor', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--random_seed', action='store_true', help='use seed randomly')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--warmup_dec_epoch', type=int, default=9, help='warmup decision epoch')
parser.add_argument('--decision_freq', type=int, default=10, help='decision freq epoch')
parser.add_argument('--history_size', type=int, default=4, help='number of stored epoch scores')
parser.add_argument('--use_history', action='store_true', help='use history for decision')
parser.add_argument('--in_channels', default=128, type=int, help='the channel size of input point cloud ')
parser.add_argument('--post_val', action='store_true', default=False, help='validate after each decision')
parser.add_argument('--K', default=9, help='use to search')

args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.save = 'log/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# writer = SummaryWriter(log_dir=args.save, max_queue=50)

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
    # writer.add_image(type + '_score', score_img, epoch)

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
    # 选择部分节点
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
    subgraph.val_mask =  torch.cat([subgraph.val_mask, graph.val_mask[subgraph_nodes]], dim=0)
    subgraph.test_mask =  torch.cat([subgraph.test_mask, graph.test_mask[subgraph_nodes]], dim=0)
 
    # 可选：复制节点特征等其他信息
    
    # subgraph.x = subgraph_nodes
    print("subgraph_node_after",subgraph_nodes.size())
    return subgraph
 
def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, num_splits: int = 10):

    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)

    trains, vals, tests = [], [], []

    for _ in range(num_splits):
        indices = torch.randperm(num_samples)

        train_mask = torch.zeros(num_samples, dtype=torch.bool)
        train_mask.fill_(False)
        train_mask[indices[:train_size]] = True

        test_mask = torch.zeros(num_samples, dtype=torch.bool)
        test_mask.fill_(False)
        test_mask[indices[train_size: test_size + train_size]] = True

        val_mask = torch.zeros(num_samples, dtype=torch.bool)
        val_mask.fill_(False)
        val_mask[indices[test_size + train_size:]] = True

        trains.append(train_mask.unsqueeze(1))
        vals.append(val_mask.unsqueeze(1))
        tests.append(test_mask.unsqueeze(1))

    train_mask_all = torch.cat(trains, 1)
    val_mask_all = torch.cat(vals, 1)
    test_mask_all = torch.cat(tests, 1)

    return train_mask_all, val_mask_all, test_mask_all
      
def evaluate(data,feature, edge_index, idx, label, num_classes, model):
    
    model.eval()
    # print(data.x,data.edge_index)
    eval_feature = feature
    output = model(data)
    score = output[idx]
    loss = hard_xe_loss_op(score, label)

    f1 = torch.mean(n_f1_score(torch.argmax(score, dim=1), label, num_classes=num_classes)).numpy()
    acc = accuracy_score(torch.argmax(score, dim=1).cpu(), label.cpu())
        
    return f1, acc

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

    # if args.data == 'Coauthor_Physics':
    #     dataset = Coauthor('../../data/Coauthor_Physics', 'Physics')
    # elif args.data == 'Coauthor_CS':
    #     dataset = Coauthor('../../data/Coauthor_CS', 'CS')
    dataset = Coauthor(args.root, args.data)
    
    raw_dir = dataset.raw_dir
    data = dataset[0]
    data = utils.save_load_split(data, raw_dir, 1, utils.gen_uniform_60_20_20_split)

    edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.edge_index = edge_index 

    criterion = nn.CrossEntropyLoss().cuda()
    
    subgraphs = []
    models = []
    nets = []
    genotypes = []
    accs = []
    ranks = []
    n_classes= dataset.num_classes
    for i in range(args.K):
        SAINT_data = GraphSAINT.GraphSAINTNodeSampler(data, batch_size=5000)
        subgraph = next(iter(SAINT_data))
        # unique_nodes = torch.unique(subgraph.edge_index)
        # print("original_nodes",unique_nodes.size(0))
        # print(subgraph)
        # print(data)
        model = Network(args.init_channels, dataset.num_classes, args.num_cells, criterion,
                        args.n_steps, in_channels=dataset.num_features).cuda()
    
        subgraph = subgraph.to(DEVICE)
        subgraphs.append(subgraph)
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        num_edges = model._steps * 2
        post_train = 10
        args.epochs = args.warmup_dec_epoch + args.decision_freq * (num_edges - 1) + post_train + 1
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
        logging.info('normal_selected_idxs: {}'.format(normal_selected_idxs))
        logging.info('normal_candidate_flags: {}'.format(normal_candidate_flags))
        model.normal_selected_idxs = normal_selected_idxs
        model.normal_candidate_flags = normal_candidate_flags

 
        count = 0
        normal_probs_history = []
        for epoch in range(args.epochs):
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            # training
            train_acc, train_obj  = train(subgraph, model, architect, criterion, optimizer, lr)
            valid_acc, valid_obj = infer(subgraph, model, criterion)
            test_acc, test_obj = infer(subgraph, model, criterion,test=True)
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
                    valid_acc, valid_obj = infer(subgraph, model, criterion)
                    test_acc, test_obj = infer(subgraph, model, criterion,test=True)
                    logging.info('post valid_acc %f', valid_acc)
            # writer.add_scalar('stats/train_acc', train_acc, epoch)
            # writer.add_scalar('stats/valid_acc', valid_acc, epoch)
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            scheduler.step()


        logging.info("#" * 30 + " Done " + "#" * 30)
        logging.info('genotype = %s', model.get_genotype())
        nets.append(model)
        genotype = model.get_genotype()
        genotypes.append(genotype)
        nets.append(model)
 
        
        model = Model(args.init_channels, n_classes, args.num_cells, False, genotype,
                       in_channels=dataset.num_features).cuda()
        optimizer_model = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
 
        for i in range(100):
            model.train()
            target = subgraph.y
            n = len(target)
            optimizer_model.zero_grad()
            logits = model(subgraph)[0]
            loss = criterion(logits[subgraph.train_mask], torch.squeeze(target[subgraph.train_mask]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer_model.step()
            # pdb.set_trace()
            y_pred = logits.argmax(dim=-1, keepdim=True)
            valid_acc, valid_obj = infer(subgraph, model, criterion)
            test_acc, test_obj = infer(subgraph, model, criterion,test=True)
            
            logging.info("epoch %s val_acc %s",i,valid_acc)
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
            acc = infer(batch, models[i], criterion)[0]
            num = acc*len(batch.y[batch.test_mask])
            sum += num
        acc = sum/(len(data.y[data.test_mask]))
        temp[i] = acc
  
    ranks.append(temp.argsort())  
    accs.append(temp) 
     
    for i in range(args.K):
        temp = torch.zeros(size=(args.K,))
        for j in range(args.K):
            temp [j] = infer(subgraphs[i], models[j], criterion)[0]
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
    # import pdb
    # pdb.set_trace()

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
            train_acc = train(subgraph, model, architect, criterion, optimizer, lr)
            valid_acc, valid_obj = infer(subgraph, model, criterion)
            test_acc, test_obj = infer(subgraph, model, criterion,test=True)
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
                    valid_acc, valid_obj = infer(subgraph, model, criterion)
                    test_acc, test_obj = infer(subgraph, model, criterion,test=True)
                    logging.info('post valid_acc %f', valid_acc) 
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            scheduler.step()
        logging.info("#" * 30 + " Done " + "#" * 30)
        logging.info('genotype = %s', model.get_genotype())


def train(data, model, architect, criterion, optimizer, lr):
    return train_trans(data, model, architect, criterion, optimizer, lr)

def train_trans(data, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
    mask = data.train_mask
    target = Variable(data.y[mask], requires_grad=False).to(DEVICE) 
    architect.step(data.to(DEVICE), lr, optimizer, unrolled=args.unrolled, mask=True)

    #train loss
    logits = model(data.to(DEVICE))
    input = logits[mask].to(DEVICE) 
    optimizer.zero_grad()
    loss = criterion(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    acc = logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc, loss/mask.sum().item() 

def infer( data, model, criterion, test=False):
    return infer_trans(data, model, criterion, test=test)

def infer_trans(data, model, criterion, test=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    mask = data.val_mask
    with torch.no_grad():
        logits = model(data.to(DEVICE))
    if test:
        mask = data.test_mask 
    else:
        mask = data.val_mask 
    # import pdb
    # if logits
    if type(logits) is tuple:
        input = logits[0][mask].to(DEVICE)
    else:
        input = logits[mask].to(DEVICE)
    target = data.y[mask].to(DEVICE)
    loss = criterion(input, target)
    acc = input.max(1)[1].eq(target).sum().item() / mask.sum().item()
    return acc, loss/mask.sum().item()

if __name__ == '__main__': 
    main() 
