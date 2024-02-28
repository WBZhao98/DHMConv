
import os.path as osp
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd
import os
import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected, to_networkx
from networkx.algorithms.components import is_weakly_connected

from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
from torch_scatter import scatter_add
import scipy


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


def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    if edge_weight ==None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    #添加自循环
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index

    #计算度矩阵，相当于adj的行和
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float('inf')] = 0
    p = deg_inv[row] * edge_weight#边概率


    # personalized pagerank p
    #把p构造为矩阵形式
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))#加入辅助结点
    #构造新的p p_ppr
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v

    #特征分解 得到pi
    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    print('piiiiiiiiiiiiiiiiiiiiiiii', pi)
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    print(torch.max(eig_value))

    # Note that by scaling the vectors, even the sign can change. That's why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float('inf')] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float('inf')] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0
    print('L:',L)
    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]