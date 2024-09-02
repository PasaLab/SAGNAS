import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from .torch_nn import MLP, act_layer, norm_layer
from .torch_edge import DilatedKnnGraph
from torch_geometric.utils import remove_self_loops, add_self_loops
from .message_passing import MessagePassing

class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(MRConv, self).__init__()
        self.nn = MLP([in_channels*2, out_channels], act, norm, bias)
        self.aggr = aggr

    def forward(self, x, edge_index):
        """"""
        x_j = tg.utils.scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1], dim_size=x.shape[0])
        return self.nn(torch.cat([x, x_j], dim=1))

class EdgConv(tg.nn.EdgeConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)

    def forward(self, x, edge_index):
        
        return super(EdgConv, self).forward(x, edge_index)

class GATConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, head, act='relu', norm=None, bias=True):
        super(GATConv, self).__init__()
        
        self.gconv = tg.nn.GATConv(in_channels, out_channels, head, concat=False, bias=bias, dropout=0.6)
        m =[]
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)


    def forward(self, x, edge_index):
        import pdb
        # pdb.set_trace()
        out = self.unlinear(self.gconv(x, edge_index))
        return out

class SAGEConv(tg.nn.SAGEConv):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 norm=True,
                 bias=True,
                 relative=False,
                 **kwargs):
        self.relative = relative
        if norm is not None:
            super(SAGEConv, self).__init__(in_channels, out_channels, True, bias, **kwargs)
        else:
            super(SAGEConv, self).__init__(in_channels, out_channels, False, bias, **kwargs)
        self.nn = nn
        

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=size, x=x)

    # def message(self, x_i, x_j):
    #     if self.relative:
    #         x = torch.matmul(x_j - x_i, self.weight)
    #     else:
    #         x = torch.matmul(x_j, self.weight)
    #     return x

    # def update(self, aggr_out, x):
    #     out = self.nn(torch.cat((x, aggr_out), dim=1))
    #     if self.bias is not None:
    #         out = out + self.bias
    #     if self.normalize:
    #         out = F.normalize(out, p=2, dim=-1)
    #     return out

class RSAGEConv(SAGEConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, relative=False):
        nn = MLP([out_channels + in_channels, out_channels], act, norm, bias)
        super(RSAGEConv, self).__init__(in_channels, out_channels, nn, norm, bias, relative)

class SemiGCNConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(SemiGCNConv, self).__init__()
        self.gconv = tg.nn.GCNConv(in_channels, out_channels, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out

class GinConv(tg.nn.GINConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
        super(GinConv, self).__init__(MLP([in_channels, out_channels], act, norm, bias))

    def forward(self, x, edge_index):
        return super(GinConv, self).forward(x, edge_index)

class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge',
                 act='relu', norm=None, bias=True, heads=8):
        super(GraphConv, self).__init__()
        if conv.lower() == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias)
        # elif conv.lower() == 'gat':
        #     self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        elif conv.lower() == "gat_8":
            self.gconv = GATConv(in_channels, out_channels, 8, act, norm, bias)
        elif  conv.lower() == "gat_4":
            self.gconv = GATConv(in_channels, out_channels, 4, act, norm, bias)
        elif conv.lower() == "gat":
            self.gconv = GATConv(in_channels, out_channels, 1, act, norm, bias)

        elif conv.lower() in ['gat_linear','gat_cos','gat_generalized_linear']:
            self.gconv = GeoLayer(in_channels, out_channels//heads, act, norm, bias, heads,dropout = 0.5)
        # elif conv.lower() == 'gat_cos':
        #     self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        # elif conv.lower() == 'gat_generalized_linear':
        #     self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)   
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gin':
            self.gconv = GinConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'sage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, bias, False)
        elif conv.lower() in ['sage_sum','sage_max']:
            self.gconv = GeoLayer(in_channels, out_channels, act, norm, bias, True,dropout = 0.5)
        elif conv.lower() == 'appnp':
            self.gconv = tg.nn.APPNP(K=10, alpha=0.1)
        elif conv.lower() == "arma":
            self.gconv = tg.nn.ARMAConv(in_channels, out_channels, bias=bias)
        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))

    def forward(self, x, edge_index):

        return self.gconv(x, edge_index)

class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None):
        edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)

class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
        :input: (x0, x1, x2, ... , xi), batch
        :output:(x0, x1, x2, ... , xi ,xi+1) , batch
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None):
        return self.body(x, batch) + x*self.res_scale, batch

class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(channels*2, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)

    def forward(self, x, batch=None):
        dense = self.body(x, batch)
        return torch.cat((x, dense), 1), batch

class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, channels,  conv='edge', act='relu', norm=None, bias=True, heads=8,  res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = GraphConv(channels, channels, conv, act, norm, bias, heads)
        self.res_scale = res_scale

    def forward(self, x, edge_index):
        return self.body(x, edge_index) + x*self.res_scale, edge_index

class DenseGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, in_channels,  out_channels, conv='edge', act='relu', norm=None, bias=True, heads=8):
        super(DenseGraphBlock, self).__init__()
        self.body = GraphConv(in_channels, out_channels, conv, act, norm, bias, heads)

    def forward(self, x, edge_index):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1), edge_index

class GeoLayer(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 att_type="gat",
                 agg_type="sum",
                 pool_dim=0):
        if agg_type in ["sum", "mlp"]:
            super(GeoLayer, self).__init__('add')
        elif agg_type in ["mean", "max"]:
            super(GeoLayer, self).__init__(agg_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type

        # GCN weight
        #self.gcn_weight = None

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.att_type in ["generalized_linear"]:
            self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)

        #if self.agg_type in ["mean", "max", "mlp"]:
        #    if pool_dim <= 0:
        #        pool_dim = 128
        #self.pool_dim = pool_dim
        #if pool_dim != 0:
        #    self.pool_layer = torch.nn.ModuleList()
        #    self.pool_layer.append(torch.nn.Linear(self.out_channels, self.pool_dim))
        #    self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels))
        #else:
        #    pass
        self.reset_parameters()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        if self.att_type in ["generalized_linear"]:
            glorot(self.general_att_layer.weight)

        #if self.pool_dim != 0:
        #    for layer in self.pool_layer:
        #        glorot(layer.weight)
        #        zeros(layer.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # prepare
        if torch.is_tensor(x):
            x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight).view(-1, self.heads, self.out_channels),
                 None if x[1] is None else torch.matmul(x[1], self.weight).view(-1, self.heads, self.out_channels))
        num_nodes = x.size(0) if torch.is_tensor(x) else size[0]
        return self.propagate(edge_index, size=size, x=x, num_nodes=num_nodes)

    def message(self, x_i, x_j, edge_index, num_nodes):

        if self.att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j
        #elif self.att_type == "gcn":
        #    if self.gcn_weight is None or self.gcn_weight.size(0) != x_j.size(0):  # 对于不同的图gcn_weight需要重新计算
        #        _, norm = self.norm(edge_index, num_nodes, None)
        #        self.gcn_weight = norm
        #    neighbor = self.gcn_weight.view(-1, 1, 1) * x_j
        else:
            # Compute attention coefficients.
            alpha = self.apply_attention(edge_index, num_nodes, x_i, x_j)
            alpha = softmax(alpha, edge_index[0], ptr=None, num_nodes=num_nodes)
            # Sample attention coefficients stochastically.
            if self.training and self.dropout > 0:
                alpha = F.dropout(alpha, p=self.dropout, training=True)

            neighbor = x_j * alpha.view(-1, self.heads, 1)
        #if self.pool_dim > 0:
        #    for layer in self.pool_layer:
        #        neighbor = layer(neighbor)
        return neighbor

    def apply_attention(self, edge_index, num_nodes, x_i, x_j):
        if self.att_type == "gat":
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif self.att_type == "gat_sym":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

        elif self.att_type == "linear":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            al = x_j * wl
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1)
            alpha = torch.tanh(alpha)
        elif self.att_type == "cos":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif self.att_type == "generalized_linear":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            al = x_i * wl
            ar = x_j * wr
            alpha = al + ar
            alpha = torch.tanh(alpha)
            alpha = self.general_att_layer(alpha)
        else:
            raise Exception("Wrong attention type:", self.att_type)
        return alpha

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)