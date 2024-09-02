import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import OPS, MLP
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import numpy as np


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            self._ops.append(op)

    def forward(self, x, edge_index, weights, selected_idx=None):
        
        if selected_idx is None:
                
            return sum(w * op(x, edge_index)for w, op in zip(weights, self._ops))
        else:  # unchosen operations are pruned
            return self._ops[selected_idx](x, edge_index)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
        super(Cell, self).__init__()
        self.preprocess0 = MLP([C_prev_prev, C], 'relu', 'batch', bias=False)
        self.preprocess1 = MLP([C_prev, C], 'relu', 'batch', bias=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, edge_index, weights, selected_idxs=None):
        import pdb
        #pdb.set_trace()
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        
        for i in range(self._steps):
            o_list = []
            for j, hh in enumerate(states):
                if selected_idxs[offset + j] == -1: # undecided mix edges
                    o = self._ops[offset + j](hh, edge_index, weights[offset + j])  # call the gcn module
                    o_list.append(o)
                elif selected_idxs[offset + j] == PRIMITIVES.index('none'): # pruned edges
                    continue
                else:  # decided discrete edges
                    o = self._ops[offset + j](hh, edge_index, None, selected_idxs[offset + j])
                    o_list.append(o)
            s = sum(o_list)
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)



class Network(nn.Module):
    def __init__(self, C, num_classes, num_cells, criterion, steps=4, multiplier=4, stem_multiplier=3, in_channels=3):
        super(Network, self).__init__()
        self._C = C 
        self._num_classes = num_classes 
        self._num_cells = num_cells
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._in_channels = in_channels
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            MLP([in_channels, C_curr], None, 'batch', bias=False),
        )
        # self.lin1 = nn.Linear(in_channels, C_curr)
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        for i in range(self._num_cells):
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(C_prev + 1, num_classes)

        self._initialize_alphas()

        self.normal_selected_idxs = torch.tensor(len(self.alphas_normal) * [-1], requires_grad=False, dtype=torch.int)
        self.normal_candidate_flags = torch.tensor(len(self.alphas_normal) * [True],
                                                   requires_grad=False, dtype=torch.bool)

        self.entropy = torch.zeros(size=(self._steps,))
        
    def new(self):
        model_new = Network(self._C, self._num_classes, self._num_cells, self._criterion, self._steps,
                            in_channels=self._in_channels).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        model_new.normal_selected_idxs = self.normal_selected_idxs
        return model_new

    def forward(self, input):        
        x, edge_index = input.x, input.edge_index
        s0 = s1 = self.stem(x)
        
        for i, cell in enumerate(self.cells):
            weights = []
            n = 2
            start = 0
            for _ in range(self._steps):
                end = start + n
                for j in range(start, end):
                    weights.append(F.softmax(self.alphas_normal[j], dim=-1))
                start = end
                n += 1

            selected_idxs = self.normal_selected_idxs
            
            s0, s1 = s1, cell(s0, s1, edge_index, weights, selected_idxs)
        out = self.global_pooling(s1.unsqueeze(0)).squeeze(0)
        logits = self.classifier(torch.cat((out, s1), dim=1))
        return logits

    def _loss(self, data, is_valid=True):
        logits = self(data) 
                             
        if is_valid:
            input = logits[data.val_mask].cuda()
            target = data.y[data.val_mask].cuda()
        else:
            input = logits[data.train_mask].cuda()
            target = data.y[data.train_mask].cuda()
        return self._criterion(input, target)

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        self.alphas_normal = []
        for i in range(self._steps):
            for n in range(2 + i):
                #self.alphas_normal.append(Variable(1e-3 * torch.randn(num_ops).cuda(), requires_grad=True))
                self.alphas_normal.append(Variable(torch.randn(num_ops).cuda(), requires_grad=True))
                #p = torch.tensor([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9])
                #self.alphas_normal.append(Variable(p.cuda(), requires_grad=True))


        self._arch_parameters = [
            self.alphas_normal
        ]

    def arch_parameters(self):
        return self.alphas_normal

    def check_edges(self, flags, selected_idxs):
        n = 2
        max_num_edges = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            if num_selected_edges >= max_num_edges:
                for j in range(start, end):
                    if flags[j]:
                        flags[j] = False
                        selected_idxs[j] = PRIMITIVES.index('none') # pruned edges
                        self.alphas_normal[j].requires_grad = False
                    else:
                        pass
            start = end
            n += 1

        return flags, selected_idxs

    def parse_gene(self, selected_idxs):
        gene = []
        n = 2
        start = 0
        for i in range(self._steps):
            end = start + n
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    raise Exception("Contain undecided edges")
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def parse_gene_force(self, flags, selected_idxs, alphas):
        gene = []
        n = 2
        max_num_edges = 2
        start = 0
        mat = F.softmax(torch.stack(alphas, dim=0), dim=-1).detach()
        importance = torch.sum(mat[:, 1:], dim=-1)
        masked_importance = torch.min(importance, (2 * flags.float() - 1) * np.inf)
        for _ in range(self._steps):
            end = start + n
            num_selected_edges = torch.sum(1 - flags[start:end].int())
            num_edges_to_select = max_num_edges - num_selected_edges
            if num_edges_to_select > 0:
                post_select_edges = torch.topk(masked_importance[start: end], k=num_edges_to_select).indices + start
            else:
                post_select_edges = []
            for j in range(start, end):
                if selected_idxs[j] == 0:
                    pass
                elif selected_idxs[j] == -1:
                    if num_edges_to_select <= 0:
                        raise Exception("Unknown errors")
                    else:
                        if j in post_select_edges:
                            idx = torch.argmax(alphas[j][1:]) + 1
                            gene.append((PRIMITIVES[idx], j - start))
                else:
                    gene.append((PRIMITIVES[selected_idxs[j]], j - start))
            start = end
            n += 1

        return gene

    def get_genotype(self, force=False):
        if force:
            gene_normal = self.parse_gene_force(self.normal_candidate_flags,
                                                self.normal_selected_idxs,
                                                self.alphas_normal)
        else:
            gene_normal = self.parse_gene(self.normal_selected_idxs)
        n = 2
        concat = range(n + self._steps - self._multiplier, self._steps + n)
        genotype = Genotype(normal=gene_normal, normal_concat=concat)
        return genotype

    def expand(self):
        select_node = torch.argmax(self.entropy)
        self.entropy[select_node] /= 2
        p = []
        flags = []
        alphas = []
        num_ops = len(PRIMITIVES)
        begin = 0
        for i in range(self._steps):

            temp = self.normal_selected_idxs[begin:begin+i+2].clone()
            flag = self.normal_candidate_flags[begin:begin+i+2].clone()
            alpha = self.alphas_normal[begin:begin+i+2]
            begin += i+2

            if i <= select_node:
                p.append(temp)
                flags.append(flag)
                alphas.append(alpha)
            else:
                p.append(torch.cat([temp[0:select_node+2], torch.tensor([0], requires_grad=False, dtype=torch.int).cuda(), temp[select_node+2:]]))
                flags.append(torch.cat([flag[0:select_node+2], torch.tensor([False], requires_grad=False, dtype=torch.bool).cuda(), flag[select_node+2:]]))
                alphas.append(alpha[0:select_node+2]+[Variable(torch.randn(num_ops).cuda(), requires_grad=False)]+alpha[select_node+2:])

            if i == select_node:
                temp = torch.tensor((i+3) * [-1], requires_grad=False, dtype=torch.int).cuda()
                flag = torch.tensor((i+3) * [True], requires_grad=False, dtype=torch.bool).cuda()
                alpha = []
                for j in range(i+3):
                    alpha.append(Variable(torch.randn(num_ops).cuda(), requires_grad=True))
                p.append(temp)
                flags.append(flag)
                alphas.append(alpha)
        new_select = p[0]
        new_candidate = flags[0]
        new_alphas = alphas[0]
        import pdb
        for i in range(len(p)-1):
            new_select = torch.cat([new_select, p[i+1]])
            new_candidate = torch.cat([new_candidate, flags[i+1]])
            new_alphas += alphas[i+1]
        self.normal_selected_idxs = new_select.cuda()
        self.normal_candidate_flags = new_candidate.cuda()
        self.alphas_normal = new_alphas
        self._steps += 1
        self.entropy = torch.cat([self.entropy[0:select_node+1], torch.zeros(size=(1,)), self.entropy[select_node+1:]])
        return
