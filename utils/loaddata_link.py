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

    # mask_new = np.random.uniform(0, 1, num)
    # train_mask = (mask_new <= (0 + 0.8))
    # test_mask = (mask_new > (0 + 0.8))
    #
    # train_id = np.where(train_mask==True)[0]
    # test_id=np.where(test_mask==True)[0]
    # 编号 0~num
    # a = list(range(num))
    # b = list(encodeset)
    # c = zip(b, a)
    # map = dict(c)


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
        edge_dict[i].append(j)

    for i, j in zip(test_data[0], test_data[1]):
        test_dict[i].append(j)
        n_test_items+=1
    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix = np.zeros((num, num))
    # 构造关联矩阵
    tmpl=[]
    for i, j in zip(train_data[0], train_data[1]):
        x = i
        y = j  # 替换论文编号为[0,2707]
        user_item_matrix[x][y]=1
        train_dict[x].append(y)
        n_train_items+=1
        # 文章指向被引文章
        H_tail[y][x] = 1
        H_tail[x][x] = 1

        # 添加自循环
        H_head[x][x] = 1
        H_tail[y][y] = 1
        H_head[y][y] =1
        tmpl.append([x,y])
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


    return norm_adj_mat,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix,tmpl

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

    # val_dict = collections.defaultdict(list)
    # val_data = pd.read_csv(path + split + '/val.txt', sep='\t', header=None)

    # for i, j in zip(val_data[0], val_data[1]):
    #     val_dict[map[i]].append(map[j])

    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix=np.zeros((num,num))
    # 构造关联矩阵
    xx=0
    tmpl=[]
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
        tmpl.append([x,y])

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

    return norm_adj_mat,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix,tmpl

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
    templ=[]
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
            H_head[y][y] = 1
            templ.append([x,y])
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



    return features,norm_adj_mat,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix,templ


def generate_G_from_H(N,H_head,H_tail):
    n1 = np.sum(H_head, axis=1)
    n2 = np.sum(H_tail, axis=0)
    print(n1, n2)

    # idx = np.argwhere(np.all(H_tail[..., :] == 0, axis=0))
    # H_tail = np.delete(H_tail, idx, axis=1)
    #
    # idx = np.argwhere(np.all(H_head[..., :] == 0, axis=0))
    # H_head = np.delete(H_head, idx, axis=1)

    print(H_tail.shape)
    print(H_head.shape)

    D_e_head = np.sum(H_head, axis=0)
    D_e_tail = np.sum(H_tail, axis=0)

    D_v_head = np.sum(H_head, axis=1)
    D_v_tail = np.sum(H_tail, axis=1)

    print(D_v_tail.shape)
    print(D_e_tail.shape)
    print(D_v_head, D_e_head, D_v_tail, D_e_tail)

    # D_e_tail = np.mat(np.diag(np.power(D_e_tail, -1)))
    # D_v_head = np.mat(np.diag(np.power(D_v_head, -1)))
    D_e_head = np.mat(np.diag(np.power(D_e_head, -1)))
    D_v_tail = np.mat(np.diag(np.power(D_v_tail, -1)))

    D_e_head[np.isinf(D_e_head)]=0

    W = np.ones(N)
    W = np.mat(np.diag(W))
    H_head = np.mat(H_head)
    H_tail = np.mat(H_tail)
    H_head = H_head.T

    print('h_tail', H_tail)
    print('h_head', H_head)
    print('dd_head', D_e_head)
    print('dv,_tail', D_v_tail)

    P = D_v_tail * H_tail * W * D_e_head * H_head
    print((P))

    P[np.isnan(P)] = 0

    alpha = 0.1

    p_v = np.zeros([N + 1, N + 1])  # 加入辅助结点
    # 构造新的p p_ppr
    p_v[0:N, 0:N] = (1 - alpha) * P
    p_v[N, 0:N] = 1.0 / N
    p_v[0:N, N] = alpha
    p_v[N, N] = 0.0
    print(np.sum(p_v, axis=1))

    print('p_v:', p_v)

    eig_value, left_vector = np.linalg.eig(p_v)
    left_vector = left_vector.real
    ind = np.argsort(eig_value)
    pi = left_vector[:, ind[-1]]  # choose the largest eig vector
    print('pi:', pi)
    print('max', np.argmax(eig_value))
    print(ind[-1])

    pi = pi[0:N]
    print('piiii', pi)
    pi = pi / pi.sum()
    print('piiii', pi)
    p1 = np.diag(np.power(pi, 0.5))
    p2 = np.diag(np.power(pi, -0.5))
    print('p1', p1)
    PT = P.T
    I = np.diag(np.eye(N))

    G = p1 * P * p2 + p2 * PT * p1

    print('G', G)

    print('IIIIIIII', I)
    L = 0.5 * G

    L[np.isnan(L)] = 0


    deg = np.sum(L, axis=0)

    deg = deg.A
    deg = deg.squeeze()
    print(deg.shape)

    deg1 = np.diag(np.power(deg, -0.5))
    deg2 = np.diag(np.power(deg, 0.5))
    print(deg1)
    deg1[deg1 == float('inf')] = 0
    L = deg1 * L * deg1
    L[np.isnan(L)] = 0
    print('L', L)
    return L

def sample_neg(edge_dict,all_edge,num):
    import torch
    u_i_m=torch.zeros((num,num))
    user_batch=edge_dict.keys()
    neg_dict = collections.defaultdict(list)
    i=0

    for u in user_batch:
        n = len(edge_dict[u])
        neg_items=[]
        while True:
            if len(neg_items) == n:
                break
            neg_id = np.random.randint(low=0, high=num, size=1)[0]

            if neg_id not in all_edge[u] and neg_id not in neg_items:
                neg_items.append(neg_id)

                u_i_m[u][neg_id]=-1
                i+=1
        neg_dict[u]=neg_items


    return neg_dict,u_i_m

def readdata_DIHGAE(args):

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
    templ=[]
    edge_index = [[],[]]
    for i, j in zip(train_data[0], train_data[1]):
        if args.dataset == 'cora' or (str(i) in map.keys() and str(j) in map.keys() ):
            x = map[i]
            y = map[j]  # 替换论文编号为[0,2707]
            user_item_matrix[x][y] = 1

            train_dict[x].append(y)
            n_train_items += 1
            # 文章指向被引文章

            H_head[x][y] = 1
            H_tail[x][x] = 1

            # 添加自循环
            H_head[x][x] = 1
            H_tail[y][y] = 1
            H_head[y][y] = 1
            templ.append([x,y])
            edge_index[0].append(x)
            edge_index[1].append(y)

    # try:
    #     norm_adj_mat = sp.load_npz(path + split + '/s_norm_adj_mat_d1.npz')
    #     print('already load adj matrix', norm_adj_mat.shape)
    # except Exception:
    #     norm_adj_mat = generate_G_from_H(num, H_head, H_tail)
    #     norm_adj_mat = sp.csr_matrix(norm_adj_mat)
    #     print('sparse_adj', norm_adj_mat)
    #     # np.save(self.path+ '/s_norm_adj_mat_d.npy', norm_adj_mat)
    #     sp.save_npz(path + split + '/s_norm_adj_mat_d1.npz', norm_adj_mat)
    print('n_train:',n_train_items,'n_test:',n_test_items)
    print('train_users',len(train_dict.keys()))
    print('test_users',len(test_dict.keys()))

    train_idx=np.random.choice(range(num),size=800,replace=False)
    test_idx=np.random.choice(range(num),size=200,replace=False)
    features=np.array(features)

    return features,edge_index,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix,templ

def readdata_DIHGAE_withoutfts(args):

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

    # mask_new = np.random.uniform(0, 1, num)
    # train_mask = (mask_new <= (0 + 0.8))
    # test_mask = (mask_new > (0 + 0.8))
    #
    # train_id = np.where(train_mask==True)[0]
    # test_id=np.where(test_mask==True)[0]
    # 编号 0~num
    # a = list(range(num))
    # b = list(encodeset)
    # c = zip(b, a)
    # map = dict(c)


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
        edge_dict[i].append(j)

    for i, j in zip(test_data[0], test_data[1]):
        test_dict[i].append(j)
        n_test_items+=1
    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix = np.zeros((num, num))
    templ = []
    edge_index = [[], []]
    # 构造关联矩阵
    for i, j in zip(train_data[0], train_data[1]):
        x = i
        y = j  # 替换论文编号为[0,2707]
        user_item_matrix[x][y]=1
        train_dict[x].append(y)
        n_train_items+=1
        # 文章指向被引文章
        H_tail[y][x] = 1
        H_tail[x][x] = 1

        # 添加自循环
        H_head[x][x] = 1
        H_tail[y][y] = 1
        H_head[y][y] =1
        templ.append([x, y])
        edge_index[0].append(x)
        edge_index[1].append(y)
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

    # print('H_head.shape', H_head.shape)
    # print('H_tail.shape', H_tail.shape)
    # print('num', num)
    # # print('tmpl', tmpl, tmpl.shape)
    # print('user', user_item_matrix, user_item_matrix.shape)


    return edge_index,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix,templ

def readdata_DIHGAE_withoutfts1(args):
    path = args.data_path + args.dataset
    split = args.split

    # 读取引用关系文件 ： 被引文章 文章
    raw_data_cites = pd.read_csv(path + '/' + args.dataset + '.txt', sep='\t', header=None)
    train_data = pd.read_csv(path + split + '/train.txt', sep='\t', header=None)
    test_data = pd.read_csv(path + split + '/test.txt', sep='\t', header=None)

    neg_test = {}

    encodeset = set(raw_data_cites[1]) | set(raw_data_cites[0])
    num = len(encodeset)
    print('user num: ', num)

    n_train_items, n_test_items = 0, 0

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
    # 读取test集合
    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        edge_dict[map[i]].append(map[j])

    for i, j in zip(test_data[0], test_data[1]):
        test_dict[map[i]].append(map[j])
        n_test_items += 1

    # val_dict = collections.defaultdict(list)
    # val_data = pd.read_csv(path + split + '/val.txt', sep='\t', header=None)

    # for i, j in zip(val_data[0], val_data[1]):
    #     val_dict[map[i]].append(map[j])

    # for i, j in zip(neg_test[0],neg_test[1]):
    #     neg_test_dict[i].append(j)
    user_item_matrix = np.zeros((num, num))
    # 构造关联矩阵
    xx = 0
    templ = []
    edge_index = [[], []]
    for i, j in zip(train_data[0], train_data[1]):
        x = map[i]
        y = map[j]  # 替换论文编号为[0,2707]
        user_item_matrix[x][y] = 1
        xx += 1

        train_dict[x].append(y)
        n_train_items += 1
        # 文章指向被引文章
        H_head[x][y] = 1
        H_tail[x][x] = 1

        # 添加自循环
        H_head[x][x] = 1
        H_tail[y][y] = 1
        H_head[y][y] = 1
        templ.append([x, y])
        edge_index[0].append(x)
        edge_index[1].append(y)

    try:
        norm_adj_mat = sp.load_npz(path + split + '/s_norm_adj_mat_d.npz')
        print('already load adj matrix', norm_adj_mat.shape)
    except Exception:
        norm_adj_mat = generate_G_from_H(num, H_head, H_tail)
        norm_adj_mat = sp.csr_matrix(norm_adj_mat)
        print('sparse_adj', norm_adj_mat)
        # np.save(self.path+ '/s_norm_adj_mat_d.npy', norm_adj_mat)
        sp.save_npz(path + split + '/s_norm_adj_mat_d.npz', norm_adj_mat)
    print('n_train:', n_train_items, 'n_test:', n_test_items)
    print('train_users', len(train_dict.keys()))
    print('test_users', len(test_dict.keys()))

    train_idx = np.random.choice(range(num), size=800, replace=False)
    test_idx = np.random.choice(range(num), size=200, replace=False)

    return edge_index,num,edge_dict,train_dict,test_dict,neg_test_dict,n_train_items,n_test_items,user_item_matrix,templ

