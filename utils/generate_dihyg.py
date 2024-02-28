from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import pandas as pd
import numpy as np
import collections
import scipy
# 导入数据：分隔符为空格
def sample_by_class(data,label,eachclass):
    N=range(data.shape[0])
    n=int(label.max())+1
    l=[]
    for i in range(n):
        idx = np.array(np.where(label==i))
        idx = np.squeeze(idx)
        # print(idx.shape)
        nclass = np.random.choice(idx,size=eachclass,replace=False)
        l.append(nclass)
    return l

def construct_dihg():

    # 读取数据，cora.txt.content 论文序号 特征
    raw_data = pd.read_csv('./data/cora.txt.content',sep = '\t',header = None)
    num = raw_data.shape[0] # 样本点数2708
    print(raw_data)
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b,a)
    map = dict(c)
    print(map)

    # 将词向量提取为特征,第二行到倒数第二行
    features =raw_data.iloc[:,1:-1]
    # 检查特征：共1433个特征，2708个样本点
    print(features.shape)
    labels= raw_data.iloc[:, -1]
    # 特征编码
    le = LabelEncoder()
    labels = le.fit_transform(labels).astype(np.int64)
    print(labels)

    # 读取引用关系文件 ： 被引文章 文章
    raw_data_cites = pd.read_csv('./data/cora.txt.cites',sep = '\t',header = None)
    # 分别为有向箭头两端创建关联矩阵
    H_head= np.zeros((num,num))
    H_tail= np.zeros((num,num))
    # 邻接边列表（用于无向超图）
    edge_dict = collections.defaultdict(list)
    # 构造关联矩阵
    for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
        x = map[i] ; y = map[j]  #替换论文编号为[0,2707]
        edge_dict[y].append(x)

        # 文章指向被引文章
        H_head[x][y]=1
        H_tail[y][y] = 1
        # 添加自循环
        H_head[x][x] = 1
        H_tail[x][x] = 1



    # n1=np.sum(H_head,axis=1)
    # n2=np.sum(H_tail,axis=0)
    # print(n1,n2)


    # 如果存在孤立点（也没有自循环），需要删除H中全零的列（与全部顶点对应一条超边的情况，超边减少
    idx = np.argwhere(np.all(H_tail[..., :] == 0, axis=0))
    H_tail = np.delete(H_tail, idx, axis=1)
    idx = np.argwhere(np.all(H_head[..., :] == 0, axis=0))
    H_head = np.delete(H_head, idx, axis=1)

    print(H_tail.shape)
    print(H_head.shape)

    # 分别计算De，Dv
    D_e_head=np.sum(H_head,axis=0)
    D_e_tail=np.sum(H_tail,axis=0)

    D_v_head=np.sum(H_head,axis=1)
    D_v_tail=np.sum(H_tail,axis=1)


    print(D_v_head,D_e_head,D_v_tail,D_e_tail)

    # D_e_tail=np.mat(np.diag(np.power(D_e_tail,-1)))
    # D_v_head=np.mat(np.diag(np.power(D_v_head,-1)))

    # 计算-1次方
    D_e_head=np.mat(np.diag(np.power(D_e_head,-1)))
    D_v_tail=np.mat(np.diag(np.power(D_v_tail,-1)))

    # 顶点数量
    N = features.shape[0]

    # 准备转移矩阵计算参数
    W = np.ones(H_head.shape[1])
    W = np.mat(np.diag(W)) #权重矩阵
    H_head = np.mat(H_head)#转换为matrix
    H_tail = np.mat(H_tail)
    H_head_T = H_head.T #转置

    # 计算概率转移矩阵
    P = D_v_tail * H_tail * W * D_e_head * H_head_T

    P[np.isnan(P)] = 0 #把nan替换为0
    print('P row sum:',np.sum(P, axis=1)) #每行和应为1（当前顶点到其他所有顶点的概率和）

    # 添加辅助结点
    alpha = 0.1
    p_v = np.zeros([N + 1, N + 1])  # 加入辅助结点
    # 构造新的p
    p_v[0:N, 0:N] = (1 - alpha) * P
    p_v[N, 0:N] = 1.0 / N
    p_v[0:N, N] = alpha
    p_v[N, N] = 0.0

    # 达到平稳分布时，pi*P=pi
    # 求最大特征值对应的特征向量
    eig_value, left_vector = np.linalg.eig(p_v)# 特征分解
    left_vector = left_vector.real # 取实部
    ind = np.argsort(eig_value) # 排序，得到下标的顺序
    pi = left_vector[:, ind[-1]]  # choose the largest eig vector
    print(pi)
    print('max', max(eig_value))
    # 取p_v中原来的P对应的pi部分
    pi = pi[0:N]
    pi = pi / pi.sum() #norm

    # 为计算L准备参数
    p1 = np.diag(np.power(pi, 0.5))
    p2 = np.diag(np.power(pi, -0.5))
    PT = P.T
    I = np.diag(np.eye(N + 1, N + 1))

    G = p1 * P * p2 + p2 * PT * p1
    print('G', G)
    L = 0.5 * G
    #L = I - L
    L[np.isnan(L)] = 0

    features=np.array(features)

    deg = np.sum(L, axis=1)
    deg = deg.A
    deg = deg.squeeze()
    deg1 = np.diag(np.power(deg, -0.5))
    deg2=np.diag(np.power(deg,0.5))
    print(deg1)
    deg1[deg1 == float('inf')] = 0
    L = deg1 * L * deg1
    L[np.isnan(L)] = 0
    print('L',L)


    # 每个类别抽20个，train_x为抽样下标
    train_x = sample_by_class(features, labels,20)

    train_x = np.array(train_x).flatten()
    print(train_x)
    N = np.array(range(features.shape[0]))
    print((N.shape))

    #np.savetxt('train_x.txt',train_x,fmt='%d',delimiter='\t')
    train_x = np.loadtxt('train_x.txt',dtype=int)
    test_x = np.delete(N, train_x)
    print(test_x.shape)

    return L,features,labels,edge_dict,train_x,test_x


def read_data(name,eachclass=20):
    raw_data = pd.read_csv('./data/'+name+'.content', sep='\t', header=None)

    print(raw_data)
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    if name=='citeseer': b= [str(i) for i in b]
    c = zip(b, a)
    map = dict(c)
    print(map)

    # 将词向量提取为特征,第二行到倒数第二行
    features = raw_data.iloc[:, 1:-1]
    # 检查特征：共1433个特征，2708个样本点
    print(features.shape)
    labels = raw_data.iloc[:, -1]

    raw_data_cites = pd.read_csv('./data/'+name+'.cites', sep='\t', header=None)
    features = np.array(features)

    le = LabelEncoder()
    labels = le.fit_transform(labels).astype(np.int64)
    print(labels)
    labels = np.array(labels)
    print(labels.shape)
    H_head, H_tail, edge_dict = construct_H_from_cites(name,features,raw_data_cites,map)
    # 每个类别抽20个，train_x为抽样下标
    train_x = sample_by_class(features, labels,eachclass)

    train_x = np.array(train_x).flatten()
    print(train_x)
    N = np.array(range(features.shape[0]))
    print((N.shape))

    #np.savetxt('citeseer_train_x.txt',train_x,fmt='%d',delimiter='\t')
    train_x = np.loadtxt('citeseer_train_x.txt', dtype=int)
    test_x = np.delete(N, train_x)
    return features,labels,H_head,H_tail,edge_dict,train_x,test_x
# cat_data = np.load('./data/cora_ml/raw/cora_ml.npz',allow_pickle=True)
# print(cat_data.files)
# filesname=cat_data.files
# files=[]
# # for f in filesname:
# #     print(f)
# #     ff=cat_data[f]
# #     print(ff)
# #     print(cat_data[f].shape)
#
# attr_data=cat_data['attr_data']
# idx_to_attr=cat_data['idx_to_attr']
# print(attr_data)
# print(idx_to_attr)
# 导入数据：分隔符为空格

def construct_H_from_cites(name,fts,raw_data_cites,map):
    num=fts.shape[0]
    # 创建一个规模和邻接矩阵一样大小的矩阵
    H_head = np.zeros((num, num))
    H_tail = np.zeros((num, num))
    # 创建邻接矩阵
    map1 = {}
    map2 = {}
    edge_dict = collections.defaultdict(list)
    for i, j in zip(raw_data_cites[0], raw_data_cites[1]):
        if str(i) in map.keys() and str(j) in map.keys() :
            x = map[i]
            y = map[j]  # 替换论文编号为[0,2707]
            edge_dict[y].append(x)
            if y not in map1.keys():
                map1[y] = 1
            if x not in map1.keys():
                map1[x] = 1
            H_head[x][y] = 1
            H_head[x][x] = 1
            H_tail[x][x] = 1
            H_tail[y][y] = 1  # 有引用关系的样本点之间取1
    print('dict:', len(map1))  # 共有多少票论文被引用，就有多少条超边
    print('dict:', len(map2))  # 共有多少票论文被引用，就有多少条超边

    # 查看邻接矩阵的元素和（按每列汇总）
    print('edgedict:', edge_dict)
    return H_head,H_tail,edge_dict

def generate_G_from_H(features,H_head,H_tail):
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

    W = W = np.ones(features.shape[0])
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
    N = features.shape[0]
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
# features,labels,H_head,H_tail,edge_dict=read_data('cora.txt')
# L=generate_G_from_H(features,H_head,H_tail)