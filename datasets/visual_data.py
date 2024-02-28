from datasets import load_ft
from datasets import load_ft
from utils1 import hypergraph_utils as hgut
import numpy as np
from config import get_config
cfg = get_config('config/config.yaml')
import pickle as pkl
import networkx as nx
import scipy.sparse as sp




def parse_index_file(filename):
    """
    Copied from gcn
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def load_feature_construct_H(data_dir,
                             m_prob=1,
                             K_neigs=[10],
                             is_probH=True,
                             split_diff_scale=False,
                             use_mvcnn_feature=False,
                             use_gvcnn_feature=True,
                             use_mvcnn_feature_for_structure=False,
                             use_gvcnn_feature_for_structure=True):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):#读取文件数据,对应data里面的文件后缀名
        with open("{}/ind.{}.{}".format(cfg['modelnet40_ft'], cfg['activate_dataset'], names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    #把index文件每行转为int
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(cfg['modelnet40_ft'], cfg['activate_dataset']))
    test_idx_range = np.sort(test_idx_reorder) #对index文件排序

    if cfg['activate_dataset'] == 'cora.txt':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)
    features = features.todense()

    G = nx.from_dict_of_lists(graph)
    edge_list = G.adjacency_list()#邻接表
    print("edge_list:", len(edge_list))
    degree = [0] * len(edge_list)
    if cfg['add_self_loop']:#添加指向自己的边
        for i in range(len(edge_list)):
            edge_list[i].append(i)#添加自己
            degree[i] = len(edge_list[i])#更新度
    max_deg = max(degree)
    mean_deg = sum(degree) / len(degree)
    print(f'max degree: {max_deg}, mean degree:{mean_deg}')

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # one-hot labels
    n_sample = labels.shape[0]
    n_category = labels.shape[1]
    lbls = np.zeros((n_sample,))
    if cfg['activate_dataset'] == 'cora.txt':
        n_category += 1  # one-hot labels all zero: new category
        for i in range(n_sample):
            try:
                lbls[i] = np.where(labels[i] == 1)[0]  # numerical labels
            except ValueError:  # labels[i] all zeros
                lbls[i] = n_category + 1  # new category
    else:
        for i in range(n_sample):
            lbls[i] = np.where(labels[i] == 1)[0]  # numerical labels

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    return features, lbls, idx_train, idx_val,idx_test, edge_list
