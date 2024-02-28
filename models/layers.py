import math
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import Parameter
# from torch_scatter import scatter_add
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import to_undirected, is_undirected
# from torch_geometric.nn.inits import glorot, zeros
from typing import Optional, Union, List, Tuple, Dict, Any, TYPE_CHECKING
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils import softmax, add_remaining_self_loops


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):#输入维度、输出维度
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))#weight与输入矩阵相乘，转换维度为n×outft
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))#.size(1)返回列的维度
        # .data：计算历史不被记录，
        # uniform_:从均匀分布中抽样得到的值填充:随机指定weight的值，后续再通过反向传播修正
        self.weight.data.uniform_(-stdv, stdv)
        #如果有b，则同样随机指定值
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x: torch.Tensor, G: torch.Tensor):#按照一层卷积的公式计算
        x = x.matmul(self.weight)#X×W
        if self.bias is not None:# +b
            x = x + self.bias
        x = G.matmul(x)#GXW对应卷积中的GXθ
        return x


class DIGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, cached=True,
                                  bias=True, **kwargs):
                         super(DIGCNConv, self).__init__(aggr='add', **kwargs)

                         self.in_channels = in_channels
                         self.out_channels = out_channels
                         self.improved = improved
                         self.cached = cached


class DIGCNConv(MessagePassing):
    r"""The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=True,
                 bias=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))

        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x

class HNHNConv(nn.Module):
    r"""The HNHN convolution layer proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta_v2e = nn.Linear(in_channels, out_channels, bias=bias)
        self.theta_e2v = nn.Linear(out_channels, out_channels, bias=bias)

    def sparse_dropout(sp_mat: torch.Tensor, p: float, fill_value: float = 0.0) -> torch.Tensor:
        r"""Dropout function for sparse matrix. This function will return a new sparse matrix with the same shape as the input sparse matrix, but with some elements dropped out.

        Args:
            ``sp_mat`` (``torch.Tensor``): The sparse matrix with format ``torch.sparse_coo_tensor``.
            ``p`` (``float``): Probability of an element to be dropped.
            ``fill_value`` (``float``): The fill value for dropped elements. Defaults to ``0.0``.
        """
        device = sp_mat.device
        sp_mat = sp_mat.coalesce()
        assert 0 <= p <= 1
        if p == 0:
            return sp_mat
        p = torch.ones(sp_mat._nnz(), device=device) * p
        keep_mask = torch.bernoulli(1 - p).to(device)
        fill_values = torch.logical_not(keep_mask) * fill_value
        new_sp_mat = torch.sparse_coo_tensor(
            sp_mat._indices(),
            sp_mat._values() * keep_mask + fill_values,
            size=sp_mat.size(),
            device=sp_mat.device,
            dtype=sp_mat.dtype,
        )
        return new_sp_mat

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        # v -> e
        X = self.theta_v2e(X)
        if self.bn is not None:
            X = self.bn(X)
        Y = self.act(self.v2e(X, aggr="mean"))
        # e -> v
        Y = self.theta_e2v(Y)
        X = self.e2v(Y, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.v2e_aggregation(X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update(X, e_weight)
        return X
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = self.sparse_dropout(self.H_T, drop_rate)
            else:
                P = self.H_T
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight.shape[0]
            ), "The size of v2e_weight must be equal to the size of self.v2e_weight."
            P = torch.sparse_coo_tensor(self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device)
            if drop_rate > 0.0:
                P = self.sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X
    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e, X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e, "The size of e_weight must be equal to the size of self.num_e."
            X = e_weight * X
        return X
    def e2v(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.e2v_aggregation(X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update(X)
        return X

    def e2v_aggregation(
            self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = self.sparse_dropout(self.H, drop_rate)
            else:
                P = self.H
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                    e2v_weight.shape[0] == self.e2v_weight.shape[0]
            ), "The size of e2v_weight must be equal to the size of self.e2v_weight."
            P = torch.sparse_coo_tensor(self.H._indices(), e2v_weight, self.H.shape, device=self.device)
            if drop_rate > 0.0:
                P = self.sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X
    def e2v_update(self, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        if self.device != X.device:
            self.to(X.device)
        return X

class DiHGAEConvEdge_withoutfts(MessagePassing):
    def __init__(self, out_feats, nums, attention,iterable):
        super().__init__(aggr="add")
        self.attention = attention
        self.iterable = iterable
        self.nums = nums
        self.dropout = nn.Dropout(p=0.2)
        self.leakrelu = nn.LeakyReLU(0.1)
        if self.iterable:
        #可迭代的注意力系数
            self.a = nn.Parameter(torch.ones(size=(out_feats, 1)))
        #不可迭代注意力系数（降维）
        else:
            self.a = torch.ones(size=(out_feats, 1))
        self.x = nn.Parameter(torch.empty(self.nums,out_feats))
        nn.init.xavier_uniform_(self.x)
        nn.init.xavier_uniform_(self.a)

        # self.alpha = torch.nn.Parameter(torch.ones(1))
        # self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        # 计算 Wh
        x = self.x
        x_edge = 0 * x
        x = torch.vstack((x, x_edge))
        x = self.dropout(x)

        row, col = edge_index

        # 根据超边的度计算注意力权重
        # in_degree = degree(col)
        # out_degree = degree(row)
        # alpha = 1
        # beta = 0
        #
        # in_norm_inv = pow(in_degree, -alpha)
        # out_norm_inv = pow(out_degree, -beta)
        # out_norm = out_norm_inv[row]
        # in_norm = in_norm_inv[col]
        # norm = in_norm * out_norm
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 启动消息传播
        h_prime = self.propagate(edge_index, x=x, norm=norm)
        return h_prime

    def message(self, x_j, edge_index_i, norm):
        if self.attention:
        # 注意力系数计算
            e = torch.matmul(x_j, self.a)
            e = self.leakrelu(e)
            alpha = softmax(e, edge_index_i)
            x_j = x_j * alpha
        x_j = norm.view(-1, 1) * x_j
        return x_j

class DiHGAEConvEdge(MessagePassing):
    def __init__(self, in_feats, out_feats,attention,iterable):
        super().__init__(aggr="add")
        self.attention = attention
        self.iterable = iterable
        self.lin = nn.Linear(in_feats, out_feats, bias = False)
        self.dropout = nn.Dropout(p=0.2)

        if iterable:
            self.a = nn.Parameter(torch.ones(size=(out_feats, 1)))
        else:
            #不可迭代注意力系数（降维）
            self.a = torch.ones(size=(out_feats, 1))

        self.leakrelu = nn.LeakyReLU(0.1)
        nn.init.xavier_uniform_(self.a)
        # self.alpha = torch.nn.Parameter(torch.ones(1))
        # self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        # 计算 Wh
        x_edge = 0 * x
        x = torch.vstack((x, x_edge))
        # x = self.dropout(x)
        h = self.lin(x)
        row, col = edge_index

            # 根据超边的度计算注意力权重
            # in_degree = degree(col)
            # out_degree = degree(row)
            #
            # alpha = 1
            # beta = 0
            #
            # in_norm_inv = pow(in_degree, -alpha)
            # out_norm_inv = pow(out_degree, -beta)
            #
            # out_norm = out_norm_inv[row]
            # in_norm = in_norm_inv[col]
            # norm = in_norm * out_norm
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 启动消息传播
        h_prime = self.propagate(edge_index, x=h, norm=norm)
        # h_prime = self.propagate(edge_index, x=h)
        return h_prime

    def message(self, x_j, edge_index_i,norm):
        if self.attention:
            # 注意力系数计算
            e = torch.matmul(x_j, self.a)
            e = self.leakrelu(e)
            alpha = softmax(e, edge_index_i)
            x_j = x_j * alpha
        x_j = norm.view(-1, 1) * x_j
        return x_j

    # def message(self, x_j, norm):
    #     return norm.view(-1, 1) * x_j

class DiHGAEConvNode(MessagePassing):
    def __init__(self,out_feats):
        super().__init__(aggr="add")
        # self.att = nn.Parameter(torch.ones(size=(out_feats, 1)))
        self.leakrelu = nn.LeakyReLU(0.2)
        # nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        #
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 启动消息传播
        h_prime = self.propagate(edge_index, x=x, norm=norm)
        # h_prime = self.propagate(edge_index, x=x)
        h_prime = h_prime[:int(h_prime.shape[0]/2),]
        return h_prime

    def message(self, x_j, edge_index_j, norm):
        # e = torch.matmul(x_j, self.att)
        # e = self.leakrelu(e)
        # alpha = softmax(e, edge_index_j)
        # x_j = x_j * alpha
        # print(x_j.shape)
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
        # return x_j

class DiHGAEConvNodeneg(MessagePassing):
    def __init__(self,out_feats):
        super().__init__(aggr="add")
        # self.b = nn.Parameter(torch.ones(size=(out_feats, 1)))
        self.leakrelu = nn.LeakyReLU(0.2)
        # nn.init.xavier_uniform_(self.b)

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        #
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 启动消息传播
        h_prime = self.propagate(edge_index, x=x, norm=norm)
        # h_prime = self.propagate(edge_index, x=x)
        h_prime = h_prime[:int(h_prime.shape[0]/2),]
        return h_prime

    def message(self, x_j, edge_index_j, norm):
        # e = torch.matmul(x_j, self.b)
        # e = self.leakrelu(e)
        # alpha = softmax(e, edge_index_j)
        # x_j = x_j * alpha
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
        # return x_j

class DiHGAEConvEdge_classificate(MessagePassing):
    def __init__(self, in_feats, out_feats,attention,iterable):
        super().__init__(aggr="add")
        self.attention = attention
        self.iterable = iterable
        self.lin = nn.Linear(in_feats, out_feats, bias = False)
        self.dropout = nn.Dropout(p=0.9)

        if iterable:
            self.a = nn.Parameter(torch.ones(size=(out_feats, 1)))
        else:
            #不可迭代注意力系数（降维）
            self.a = torch.ones(size=(out_feats, 1))

        self.leakrelu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.a)
        # self.alpha = torch.nn.Parameter(torch.ones(1))
        # self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        edge_index, _ = add_remaining_self_loops(edge_index)
        # 计算 Wh
        x_edge = 0 * x
        x = torch.vstack((x, x_edge))
        x = self.dropout(x)
        h = self.lin(x)
        row, col = edge_index

        # 根据超边的度计算注意力权重
        # in_degree = degree(col)
        # out_degree = degree(row)
        #
        # alpha = 1
        # beta = 0
        #
        # in_norm_inv = pow(in_degree, -alpha)
        # out_norm_inv = pow(out_degree, -beta)
        #
        # out_norm = out_norm_inv[row]
        # in_norm = in_norm_inv[col]
        # norm = in_norm * out_norm
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 启动消息传播
        h_prime = self.propagate(edge_index, x=h, norm=norm)
        return h_prime

    def message(self, x_j, edge_index_i, norm):
        if self.attention:
            # 注意力系数计算
            e = torch.matmul(x_j, self.a)
            e = self.leakrelu(e)
            alpha = softmax(e, edge_index_i)
            x_j = x_j * alpha
        x_j = norm.view(-1, 1) * x_j
        return x_j

class DiHGAEConvNode1(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, para):
        self.para = para
        edge_index, _ = add_remaining_self_loops(edge_index)
        #
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 启动消息传播
        h_prime = self.propagate(edge_index, x=x, norm=norm)
        h_prime= h_prime[:int(h_prime.shape[0]/2),]
        return h_prime

    def message(self, x_j, norm):
        x_j = torch.vstack((1/2 * x_j[:int(x_j.shape[0]/2),], self.para * x_j[int(x_j.shape[0]/2):,]))
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class DiHGAEConvNode2(MessagePassing):
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, para):
        self.para = para
        edge_index, _ = add_remaining_self_loops(edge_index)
        #
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 启动消息传播
        h_prime = self.propagate(edge_index, x=x, norm=norm)
        h_prime= h_prime[:int(h_prime.shape[0]/2),]
        return h_prime

    def message(self, x_j, norm):
        x_j = torch.vstack((1/2 * x_j[:int(x_j.shape[0]/2),], (1-self.para) * x_j[int(x_j.shape[0]/2):,]))
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j