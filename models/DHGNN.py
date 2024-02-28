import copy
from time import time

from torch import nn
from models import HGNN_conv
import torch
import torch.nn.functional as F
import numpy as np
from models import DiHGAEConvEdge,DiHGAEConvNode,DiHGAEConvEdge_withoutfts,DiHGAEConvEdge_classificate,DiHGAEConvNodeneg


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):#输入维度（特征向量维度），类别数，隐藏层维度
        super(HGNN, self).__init__()
        self.dropout = dropout
        #两层卷积 hgc为layers里的HGNN_conv类
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        #hgc1为HGNN_conv类，输入x，G，调用类内的forward函数
        x = F.relu(self.hgc1(x, G))#激活函数relu
        #x=self.hgc1(x,G)
        #x=F.tanh(x)
        #print("self.hgc1 卷积一次",x.size())
        x = F.dropout(x, self.dropout)#随机丢弃
        x = self.hgc2(x, G)
        #np.save("X_fx.npy",x.detach().numpy())
        return x


class DHGCF(nn.Module):
    def __init__(self,nums,norm_L,args):
        super(DHGCF,self).__init__()
        self.nums=nums
        self.norm_L=norm_L
        self.device = args.device
        self.emb_size = args.embed_size

        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.regs = eval(args.regs)[0]


        self.weight_dict,self.embedding_dict = self.init_weight()
        # initializer = nn.init.xavier_uniform_
        # self.embedding_dict=initializer(torch.empty(self.nums,
        #                                           self.emb_size))
        self.norm_L = torch.tensor(norm_L.todense(),dtype=torch.float32).to(self.device)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(norm_L).to(self.device)


    def init_weight(self):
        # xavier init
        # 一种神经网络初始化方法，主要的目标就是使得每一层输出的方差应该尽量相等
        # uniform为均匀分布
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(self.nums,
                                                 self.emb_size)))
        })

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers# layers 为每一层的输出大小  layers=[embsize,64,64,64]
        for k in range(len(self.layers)):# 对应每一层的输出。更新可学习权重矩阵的大小
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            # weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
            #                                                           layers[k+1])))})


        return  weight_dict,embedding_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_loss(self,u_g_embeddings,pred_p,pred_n):

        # pos_loss = -torch.log(pred_p + regs).mean()
        # neg_loss = -torch.log(1 - pred_n + regs).mean()
        # mf_loss = pos_loss + neg_loss
        maxi = nn.LogSigmoid()(pred_p - pred_n)
        mf_loss = -1 * torch.mean(maxi)

        regularizer = (torch.norm(u_g_embeddings) ** 2) / 2
        emb_loss = self.decay * regularizer / self.nums
        batch_loss = mf_loss

        return batch_loss,mf_loss,emb_loss

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)
        #maxi=torch.nn.functional.softplus(pos_scores - neg_scores)
        #mf_loss =  torch.mean(maxi)
        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2 ) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def hgconv(self,embedding,adj,k):
        sum_embeddings = torch.matmul(embedding, self.weight_dict['W_gc_%d' % k]) \
                         + self.weight_dict['b_gc_%d' % k]
        return adj.matmul(sum_embeddings)

    def forward(self, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,#正则化拉普拉斯
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = self.embedding_dict['emb']

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)# 拉普拉斯 L* embddding

            # transformed sum messages of neighbors.

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            # ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings )
            #ego_embeddings = nn.functional.relu(sum_embeddings)
            ego_embeddings=sum_embeddings

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]#拼接每一层的embedding


        all_embeddings = torch.cat(all_embeddings, 1)#纵向拼接

        i_g_embeddings = all_embeddings


        """
        *********************************************************
        look up.
        """

        return i_g_embeddings


class DHGCF1(nn.Module):
    def __init__(self,fts,nums,norm_L,args):
        super(DHGCF1,self).__init__()
        self.nums=nums
        self.norm_L=norm_L
        self.device = args.device
        self.emb_size = args.embed_size

        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.regs = eval(args.regs)[0]
        self.fts=fts


        self.weight_dict = self.init_weight()
        # initializer = nn.init.xavier_uniform_
        # self.embedding_dict=initializer(torch.empty(self.nums,
        #                                           self.emb_size))
        self.norm_L = torch.tensor(norm_L.todense(),dtype=torch.float32).to(self.device)
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(norm_L).to(self.device)


    def init_weight(self):
        # xavier init
        # 一种神经网络初始化方法，主要的目标就是使得每一层输出的方差应该尽量相等
        # uniform为均匀分布
        initializer = nn.init.xavier_uniform_



        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers# layers 为每一层的输出大小  layers=[embsize,64,64,64]

        weight_dict.update({'W_gc_%d'%0: nn.Parameter(initializer(torch.empty(self.fts.shape[1],
                                                                  layers[0])))})
        weight_dict.update({'b_gc_%d'%0: nn.Parameter(initializer(torch.empty(1, layers[0])))})
        weight_dict.update({'W_gc_%d' % 1: nn.Parameter(initializer(torch.empty(layers[0],self.fts.shape[1]
                                                                                )))})
        weight_dict.update({'b_gc_%d' % 1: nn.Parameter(initializer(torch.empty(1, self.fts.shape[1])))})
        # weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
        #                                                           layers[k+1])))})

        return  weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_loss(self,u_g_embeddings,pred_p,pred_n):


        # pos_loss = -torch.log(pred_p + regs).mean()
        # neg_loss = -torch.log(1 - pred_n + regs).mean()
        # mf_loss = pos_loss + neg_loss
        maxi = nn.LogSigmoid()(pred_p - pred_n)
        mf_loss = -1 * torch.mean(maxi)


        batch_loss = mf_loss

        return batch_loss,mf_loss
    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())
    def hgconv(self,embedding,adj,k):
        sum_embeddings = torch.matmul(embedding, self.weight_dict['W_gc_%d' % k]) \
                         + self.weight_dict['b_gc_%d' % k]
        return adj.matmul(sum_embeddings)
    def forward1(self,user,pos_items,neg_items,drop_flag=True):
        ego_embeddings=self.embedding_dict['emb']

        ego_embeddings=self.hgconv(ego_embeddings,self.norm_L,0)
        ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(ego_embeddings)
        ego_embeddings1 = F.dropout(ego_embeddings,0.7)

        ego_embeddings2 = self.hgconv(ego_embeddings1, self.norm_L, 1)
        ego_embeddings=torch.cat((ego_embeddings1,ego_embeddings2),dim=0)


        return ego_embeddings,ego_embeddings,[]



    def forward(self, drop_flag=True):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,#正则化拉普拉斯
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj

        ego_embeddings = self.fts

        all_embeddings = [ego_embeddings]


        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)# 拉普拉斯 L* embddding

            # transformed sum messages of neighbors.

            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            #ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings )
            # ego_embeddings = nn.functional.relu(sum_embeddings)
            ego_embeddings=sum_embeddings

            # message dropout.
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)

            # normalize the distribution of embeddings.
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            ego_embeddings= norm_embeddings

           # all_embeddings += [norm_embeddings]#拼接每一层的embedding


        #all_embeddings = torch.cat(all_embeddings, 1)#纵向拼接

        # i_g_embeddings = all_embeddings


        """
        *********************************************************
        look up.
        """

        return ego_embeddings

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat,n_hid, nclass,dropout=0.5):
        super(SGC, self).__init__()
        self.dropout = dropout

        self.W1=nn.Linear(nfeat,n_hid)

        self.W = nn.Linear(n_hid, nclass)


    def forward(self, x):
        x=self.W1(x)
        x = F.dropout(x, self.dropout)#随机丢弃
        return self.W(x)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class DirectedHGAE(torch.nn.Module):
    def __init__(self, fts, nums, edge_index, args, B_Aggre, B_Broad, attention, iterable):
        super(DirectedHGAE, self).__init__()

        in_feats = fts.shape[1]
        out_feats = 64

        self.nums = nums
        self.B_Agree = B_Aggre
        self.B_Broad = B_Broad

        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = DiHGAEConvEdge(in_feats=in_feats,out_feats=out_feats,attention=attention,iterable=iterable)
        self.conv2 = DiHGAEConvNode(out_feats=out_feats)
        self.conv3 = DiHGAEConvNodeneg(out_feats=out_feats)
        # self.conv3 = DiHGAEConvNode()
        # self.conv3 = DiHGAEConvEdge(in_feats=hidden_feats, out_feats=hidden_feats)
        # self.conv4 = DiHGAEConvNode(in_feats=hidden_feats, out_feats=out_feats)

        #超边至右侧顶点
        self.edge_index_hypergraph = []
        self.edge_index_hypergraph.append([x for x in range(nums, 2 * nums)])
        self.edge_index_hypergraph.append([x for x in range(0, nums)])
        self.edge_index_hypergraph = np.array(self.edge_index_hypergraph)

        #左侧顶点至超边
        self.edge_index_nodetoedge = edge_index
        self.edge_index_nodetoedge = np.array(self.edge_index_nodetoedge)
        self.edge_index_nodetoedge[1] = self.edge_index_nodetoedge[1] + self.nums

        if self.B_Agree:
        # 添加右侧顶点（可注释）
            self.edgetoedge = self.edge_index_hypergraph.tolist()
            self.edgetoedge[0], self.edgetoedge[1] = self.edgetoedge[1], self.edgetoedge[0]
            self.edgetoedge = np.array(self.edgetoedge)
            self.edge_index_nodetoedge = np.concatenate((self.edge_index_nodetoedge, self.edgetoedge), axis=1)

        #超边至左侧顶点
        self.edge_index_edgetonode = self.edge_index_nodetoedge.tolist()
        self.edge_index_edgetonode[0],self.edge_index_edgetonode[1] = self.edge_index_edgetonode[1],self.edge_index_edgetonode[0]
        self.edge_index_edgetonode = np.array(self.edge_index_edgetonode)

        self.edge_index_nodetoedge = torch.tensor(self.edge_index_nodetoedge, dtype=torch.long)
        self.edge_index_edgetonode = torch.tensor(self.edge_index_edgetonode, dtype=torch.long)
        self.edge_index_hypergraph = torch.tensor(self.edge_index_hypergraph, dtype=torch.long)

        self.device = args.device
        self.emb_size = args.embed_size

        self.alpha = torch.nn.Parameter((torch.FloatTensor([1.0])).to(self.device))
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.regs = eval(args.regs)[0]
        self.fts = fts

    def forward(self):
        # 顶点信息传播至超边
        x = self.conv1(self.fts, self.edge_index_nodetoedge)
        #x =  self.conv1(self.edge_index_nodetoedge)
        # 超边信息传播至顶点
        # x = F.dropout(x, p=0.5, training=self.training)
        x_1 = self.conv2(x, self.edge_index_hypergraph)
        x_2 = self.conv3(x, self.edge_index_edgetonode)

        if self.B_Broad:
            x = self.alpha * x_1 + x_2
            # x = x_1 + x_2
        else:
            x = x_1
            # x = x_2
        return x

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

class DirectedHGAE_withoutfts(torch.nn.Module):
    def __init__(self, nums, edge_index, args, B_Aggre, B_Broad, attention, iterable):
        super(DirectedHGAE_withoutfts, self).__init__()

        out_feats = 64

        self.nums = nums
        self.B_Agree = B_Aggre
        self.B_Broad = B_Broad

        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = DiHGAEConvEdge_withoutfts(out_feats=out_feats,nums = self.nums, attention = attention, iterable=iterable)
        self.conv2 = DiHGAEConvNode()
        # self.conv3 = DiHGAEConvNode()
        # self.conv3 = DiHGAEConvEdge(in_feats=hidden_feats, out_feats=hidden_feats)
        # self.conv4 = DiHGAEConvNode(in_feats=hidden_feats, out_feats=out_feats)

        #超边至右侧顶点
        self.edge_index_hypergraph = []
        self.edge_index_hypergraph.append([x for x in range(nums, 2 * nums)])
        self.edge_index_hypergraph.append([x for x in range(0, nums)])
        self.edge_index_hypergraph = np.array(self.edge_index_hypergraph)

        #左侧顶点至超边
        self.edge_index_nodetoedge = edge_index
        self.edge_index_nodetoedge = np.array(self.edge_index_nodetoedge)
        self.edge_index_nodetoedge[1] = self.edge_index_nodetoedge[1] + self.nums

        if self.B_Agree:
        # 添加右侧顶点（可注释）
            self.edgetoedge = self.edge_index_hypergraph.tolist()
            self.edgetoedge[0], self.edgetoedge[1] = self.edgetoedge[1], self.edgetoedge[0]
            self.edgetoedge = np.array(self.edgetoedge)
            self.edge_index_nodetoedge = np.concatenate((self.edge_index_nodetoedge, self.edgetoedge), axis=1)

        #超边至左侧顶点
        self.edge_index_edgetonode = self.edge_index_nodetoedge.tolist()
        self.edge_index_edgetonode[0],self.edge_index_edgetonode[1] = self.edge_index_edgetonode[1],self.edge_index_edgetonode[0]
        self.edge_index_edgetonode = np.array(self.edge_index_edgetonode)

        self.edge_index_nodetoedge = torch.tensor(self.edge_index_nodetoedge, dtype=torch.long)
        self.edge_index_edgetonode = torch.tensor(self.edge_index_edgetonode, dtype=torch.long)
        self.edge_index_hypergraph = torch.tensor(self.edge_index_hypergraph, dtype=torch.long)

        self.device = args.device
        self.emb_size = args.embed_size

        self.alpha = torch.nn.Parameter((torch.FloatTensor([1.0])).to(self.device))
        # self.beta = torch.nn.Parameter((torch.FloatTensor([1.0])).to(self.device))
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.regs = eval(args.regs)[0]

    def forward(self):
        # 顶点信息传播至超边
        x = self.conv1(self.edge_index_nodetoedge)
        # 超边信息传播至顶点
        x_1 = self.conv2(x, self.edge_index_hypergraph)
        x_2 = self.conv2(x, self.edge_index_edgetonode)
        # print(self.alpha)
        if self.B_Broad:
            x = self.alpha * x_1 + x_2
        else:
            x = x_1
            # x = x_2
        return x

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

class DirectedHGAE_classificate(torch.nn.Module):
    def __init__(self, fts, nums, edge_index, args, nclass, B_Aggre, B_Broad, attention, iterable):
        super(DirectedHGAE_classificate, self).__init__()

        in_feats= fts.shape[1]
        out_feats = 64
        self.nclass = nclass

        self.nums = nums
        self.B_Agree = B_Aggre
        self.B_Broad = B_Broad

        self.dropout = nn.Dropout(p=0.2)
        self.lin = nn.Linear(out_feats, self.nclass, bias=True)
        self.conv1 = DiHGAEConvEdge_classificate(in_feats=in_feats,out_feats=out_feats,attention=attention,iterable=iterable)
        # self.conv1 = DiHGAEConvEdge_withoutfts(in_feats=in_feats, out_feats=out_feats,nums = self.nums)
        self.conv2 = DiHGAEConvNode(out_feats=nclass)
        # self.conv3 = DiHGAEConvNode()
        # self.conv3 = DiHGAEConvEdge(in_feats=hidden_feats, out_feats=hidden_feats)
        # self.conv4 = DiHGAEConvNode(in_feats=hidden_feats, out_feats=out_feats)

        #超边至右侧顶点
        self.edge_index_hypergraph = []
        self.edge_index_hypergraph.append([x for x in range(nums, 2 * nums)])
        self.edge_index_hypergraph.append([x for x in range(0, nums)])
        self.edge_index_hypergraph = np.array(self.edge_index_hypergraph)

        #左侧顶点至超边
        self.edge_index_nodetoedge = edge_index
        self.edge_index_nodetoedge = np.array(self.edge_index_nodetoedge)
        self.edge_index_nodetoedge[1] = self.edge_index_nodetoedge[1] + self.nums

        if self.B_Agree:
            # 添加右侧顶点（可注释）
            self.edgetoedge = self.edge_index_hypergraph.tolist()
            self.edgetoedge[0], self.edgetoedge[1] = self.edgetoedge[1], self.edgetoedge[0]
            self.edgetoedge = np.array(self.edgetoedge)
            self.edge_index_nodetoedge = np.concatenate((self.edge_index_nodetoedge, self.edgetoedge), axis=1)

        #超边至左侧顶点
        self.edge_index_edgetonode = self.edge_index_nodetoedge.tolist()
        self.edge_index_edgetonode[0],self.edge_index_edgetonode[1] = self.edge_index_edgetonode[1],self.edge_index_edgetonode[0]
        self.edge_index_edgetonode = np.array(self.edge_index_edgetonode)

        self.edge_index_nodetoedge = torch.tensor(self.edge_index_nodetoedge, dtype=torch.long)
        self.edge_index_edgetonode = torch.tensor(self.edge_index_edgetonode, dtype=torch.long)
        self.edge_index_hypergraph = torch.tensor(self.edge_index_hypergraph, dtype=torch.long)

        self.device = args.device
        self.emb_size = args.embed_size

        self.alpha = torch.nn.Parameter((torch.FloatTensor([1.0])).to(self.device))
        # self.beta = torch.nn.Parameter((torch.FloatTensor([1.0])).to(self.device))
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout

        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]
        self.regs = eval(args.regs)[0]
        self.fts = fts

    def forward(self):
        # 顶点信息传播至超边
        x = self.conv1(self.fts,self.edge_index_nodetoedge)
        #x =  self.conv1(self.edge_index_nodetoedge)
        # 超边信息传播至顶点
        # x = F.dropout(x, p=0.5, training=self.training)
        x_1 = self.conv2(x, self.edge_index_hypergraph)
        x_2 = self.conv2(x, self.edge_index_edgetonode)


        if self.B_Broad:
            x = self.alpha * x_1 + x_2
        else:
            x = x_1
            # x = x_2
        x = self.dropout(x)
        x = self.lin(x)
        return x

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())