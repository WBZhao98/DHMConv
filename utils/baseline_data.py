import collections
import numpy as np
import pandas as pd
import scipy.sparse as sp
import multiprocessing


import itertools
from functools import partial

def readdata(args):
    path=args.data_path + args.dataset
    split=args.split



    # 读取引用关系文件 ： 被引文章 文章
    raw_data_cites = pd.read_csv(path +'/'+ args.dataset +'.txt', sep='\t', header=None)
    train_data = pd.read_csv(path+split+'/train.txt', sep='\t', header=None)
    test_data = pd.read_csv(path+split+'/test.txt', sep='\t', header=None)
    #neg_test = pd.read_csv(path+'/test_neg_edges.txt', sep='\t', header=None)
    neg_test={}
    num = max(max(raw_data_cites[0]), max(raw_data_cites[1]))

    #encodeset = set(raw_data_cites[1]) | set(raw_data_cites[0])
    num += 1
    print('user num: ',num)

    n_train_items,n_test_items =0,0




    # 邻接边列表（用于无向超图）
    edge_dict = collections.defaultdict(list)
    train_dict = collections.defaultdict(list)
    test_dict = collections.defaultdict(list)
    neg_test_dict = collections.defaultdict(list)
    #读取test集合
    for i,j in zip(raw_data_cites[0],raw_data_cites[1]):
        edge_dict[i].append(j)

    for i, j in zip(test_data[0], test_data[1]):
        test_dict[i].append(j)
        n_test_items+=1
    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix = np.zeros((num, num))
    # 构造关联矩阵
    for i, j in zip(train_data[0], train_data[1]):
        x = i
        y = j  # 替换论文编号为[0,2707]
        user_item_matrix[x][y]=1
        train_dict[x].append(y)
        n_train_items+=1
        # 文章指向被引文章

    try:
        norm_adj_mat = sp.load_npz(path + split + '/s_norm_adj_mat_d.npz')
        print('already load adj matrix', norm_adj_mat.shape)
    except Exception:
        norm_adj_mat = generate_G_from_H(num, H_head, H_tail)
        norm_adj_mat = sp.csr_matrix(norm_adj_mat)
        print('sparse_adj', norm_adj_mat)
        # np.save(self.path+ '/s_norm_adj_mat_d.npy', norm_adj_mat)
        sp.save_npz(path + split + '/s_norm_adj_mat_d.npz', norm_adj_mat)
    print('n_train:',n_train_items,'n_test:',n_test_items)
    print('train_users',len(train_dict.keys()))
    print('test_users',len(test_dict.keys()))

    train_idx=np.random.choice(range(num),size=800,replace=False)
    test_idx=np.random.choice(range(num),size=200,replace=False)



    return norm_adj_mat,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix

def readdata1(args):
    path=args.data_path + args.dataset
    split=args.split



    # 读取引用关系文件 ： 被引文章 文章
    raw_data_cites = pd.read_csv(path +'/'+ args.dataset +'.txt', sep='\t', header=None)
    train_data = pd.read_csv(path+split+'/train.txt', sep='\t', header=None)
    test_data = pd.read_csv(path+split+'/test.txt', sep='\t', header=None)
    neg_test = {}

    encodeset = set(raw_data_cites[1]) | set(raw_data_cites[0])
    num=len(encodeset)
    print('user num: ',num)

    n_train_items,n_test_items =0,0



    # mask_new = np.random.uniform(0, 1, num)
    # train_mask = (mask_new <= (0 + 0.8))
    # test_mask = (mask_new > (0 + 0.8))
    #
    # train_id = np.where(train_mask==True)[0]
    # test_id=np.where(test_mask==True)[0]
    # 编号 0~num
    a = list(range(num))
    b = list(encodeset)
    c = zip(b, a)
    map = dict(c)


    # 分别为有向箭头两端创建关联矩阵
    H_head = np.diag(np.ones(num))
    H_tail = np.diag(np.ones(num))
    # 邻接边列表（用于无向超图）
    edge_dict = collections.defaultdict(list)
    train_dict = collections.defaultdict(list)
    test_dict = collections.defaultdict(list)
    neg_test_dict = collections.defaultdict(list)
    #读取test集合
    for i,j in zip(raw_data_cites[0],raw_data_cites[1]):
        edge_dict[map[i]].append(map[j])

    for i, j in zip(test_data[0], test_data[1]):
        test_dict[map[i]].append(map[j])
        n_test_items+=1
    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix=np.zeros((num,num))
    # 构造关联矩阵
    xx=0
    for i, j in zip(train_data[0], train_data[1]):
        x = map[i]
        y = map[j]  # 替换论文编号为[0,2707]
        user_item_matrix[x][y]=1
        xx+=1

        train_dict[x].append(y)
        n_train_items+=1
        # 文章指向被引文章
        H_head[x][y] = 1
        H_tail[x][x] = 1

        # 添加自循环
        H_head[x][x] = 1
        H_tail[y][y] = 1
        H_head[y][y] =1

    try:
        norm_adj_mat = sp.load_npz(path + split + '/s_norm_adj_mat_d.npz')
        print('already load adj matrix', norm_adj_mat.shape)
    except Exception:
        norm_adj_mat = generate_G_from_H(num, H_head, H_tail)
        norm_adj_mat = sp.csr_matrix(norm_adj_mat)
        print('sparse_adj', norm_adj_mat)
        # np.save(self.path+ '/s_norm_adj_mat_d.npy', norm_adj_mat)
        sp.save_npz(path + split + '/s_norm_adj_mat_d.npz', norm_adj_mat)
    print('n_train:',n_train_items,'n_test:',n_test_items)
    print('train_users',len(train_dict.keys()))
    print('test_users',len(test_dict.keys()))

    train_idx=np.random.choice(range(num),size=800,replace=False)
    test_idx=np.random.choice(range(num),size=200,replace=False)



    return norm_adj_mat,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix

def readdata_with_fts(args):
    path=args.data_path + args.dataset
    split=args.split



    # 读取引用关系文件 ： 被引文章 文章
    raw_data = pd.read_csv('./data/'+args.dataset+'.content', sep='\t', header=None)
    num = raw_data.shape[0]  # 样本点数2708
    raw_data_cites = pd.read_csv(path +'/'+ args.dataset +'.cites', sep='\t', header=None)
    train_data = pd.read_csv(path+split+'/train.txt', sep='\t', header=None)
    test_data = pd.read_csv(path+split+'/test.txt', sep='\t', header=None)

    print('user num: ',num)
    print(raw_data)

    n_train_items,n_test_items =0,0




    a = list(raw_data.index)
    b = list(raw_data[0])
    if args.dataset == 'citeseer': b = [str(i) for i in b]
    c = zip(b, a)
    map = dict(c)
    features = raw_data.iloc[:, 1:-1]


    # 分别为有向箭头两端创建关联矩阵
    H_head = np.diag(np.ones(num))
    H_tail = np.diag(np.ones(num))
    # 邻接边列表（用于无向超图）
    edge_dict = collections.defaultdict(list)
    train_dict = collections.defaultdict(list)
    test_dict = collections.defaultdict(list)
    neg_test_dict = collections.defaultdict(list)
    #读取test集合
    for i,j in zip(raw_data_cites[0],raw_data_cites[1]):
        if args.dataset == 'cora' or (str(i) in map.keys() and str(j) in map.keys() ):
            edge_dict[map[i]].append(map[j])


    for i, j in zip(test_data[0], test_data[1]):
        if args.dataset == 'cora' or (str(i) in map.keys() and str(j) in map.keys() ):
            test_dict[map[i]].append(map[j])
            n_test_items+=1
    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix=np.zeros((num,num))
    # 构造关联矩阵
    for i, j in zip(train_data[0], train_data[1]):
        if args.dataset == 'cora' or (str(i) in map.keys() and str(j) in map.keys() ):
            x = map[i]
            y = map[j]  # 替换论文编号为[0,2707]
            user_item_matrix[x][y]=1

            train_dict[x].append(y)
            n_train_items+=1
            # 文章指向被引文章
            H_head[x][y] = 1
            H_tail[x][x] = 1

            # 添加自循环
            H_head[x][x] = 1
            H_tail[y][y] = 1
            H_head[y][y] =1
    try:
        norm_adj_mat = sp.load_npz(path + split + '/s_norm_adj_mat_d1.npz')
        print('already load adj matrix', norm_adj_mat.shape)
    except Exception:
        norm_adj_mat = generate_G_from_H(num, H_head, H_tail)
        norm_adj_mat = sp.csr_matrix(norm_adj_mat)
        print('sparse_adj', norm_adj_mat)
        # np.save(self.path+ '/s_norm_adj_mat_d.npy', norm_adj_mat)
        sp.save_npz(path + split + '/s_norm_adj_mat_d1.npz', norm_adj_mat)
    print('n_train:',n_train_items,'n_test:',n_test_items)
    print('train_users',len(train_dict.keys()))
    print('test_users',len(test_dict.keys()))

    train_idx=np.random.choice(range(num),size=800,replace=False)
    test_idx=np.random.choice(range(num),size=200,replace=False)
    features=np.array(features)



    return features,norm_adj_mat,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix