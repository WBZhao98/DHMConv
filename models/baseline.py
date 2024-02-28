from torch import nn
from models import HGNN_conv
from models import DIGCNConv
import torch
import torch.nn.functional as F
from models import HNHNConv

class HGNN(nn.Module):
    def __init__(self, nums,in_ch, layer, dropout=0.5):#输入维度（特征向量维度），类别数，隐藏层维度
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.layers = eval(layer)
        initializer = nn.init.xavier_uniform_

        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_ch)))
        })

        #两层卷积 hgc为layers里的HGNN_conv类
        self.hgc1 = HGNN_conv(in_ch, self.layers[0])
        self.hgc2 = HGNN_conv(self.layers[0], self.layers[1])

    def forward(self, G, x=[]):

        if x==[]:

            x=self.embedding_dict['emb']
        #hgc1为HGNN_conv类，输入x，G，调用类内的forward函数
        x = F.relu(self.hgc1(x, G))#激活函数relu
        #x=self.hgc1(x,G)
        #x=F.tanh(x)
        #print("self.hgc1 卷积一次",x.size())
        x = F.dropout(x, self.dropout)#随机丢弃
        x = self.hgc2(x, G)
        #np.save("X_fx.npy",x.detach().numpy())
        return x


import dhg
from dhg.structure.graphs import Graph
from dhg.nn import HGNNConv


class HGNN1(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        nums:int,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        initializer = nn.init.xavier_uniform_

        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """

        X = self.embedding_dict['emb']

        for layer in self.layers:
            X = layer(X, hg)
        return X



from dhg.nn import HNHNConv
class HNHN(nn.Module):
    r"""The HNHN model proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        nums: int,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HNHNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HNHNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        initializer = nn.init.xavier_uniform_

        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """

        X = self.embedding_dict['emb']

        for layer in self.layers:
            X = layer(X, hg)
        return X

from dhg.nn import GCNConv
class GCN(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self,
                 nums:int,
                 in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        X = self.embedding_dict['emb']
        for layer in self.layers:
            X = layer(X, g)
        return X

from dhg.nn import GATConv, MultiHeadWrapper
class GAT(nn.Module):
    r"""The GAT model proposed in `Graph Attention Networks <https://arxiv.org/pdf/1710.10903>`_ paper (ICLR 2018).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        nums:int,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_heads: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            GATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        # The original implementation has applied activation layer after the final layer.
        # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = GATConv(
            hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        X = self.embedding_dict['emb']
        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, g=g)
        X = self.drop_layer(X)
        X = self.out_layer(X, g)
        return X

from dhg.nn import GraphSAGEConv

class GraphSAGE(nn.Module):
    r"""The GraphSAGE model proposed in `Inductive Representation Learning on Large Graphs <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_ paper (NIPS 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``aggr`` (``str``): The neighbor aggregation method. Currently, only mean aggregation is supported. Defaults to "mean".
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): The dropout probability. Defaults to 0.5.
    """

    def __init__(
        self,nums:int,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        aggr: str = "mean",
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGEConv(in_channels, hid_channels, aggr=aggr, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GraphSAGEConv(hid_channels, num_classes, aggr=aggr, use_bn=use_bn, is_last=True))
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, g: "Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        X = self.embedding_dict['emb']
        for layer in self.layers:
            X = layer(X, g)
        return X


from dhg.nn import HyperGCNConv
class HyperGCN(nn.Module):
    r"""The HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
            self,nums:int,
            in_channels: int,
            hid_channels: int,
            num_classes: int,
            use_mediator: bool = False,
            use_bn: bool = False,
            fast: bool = True,
            drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.layers = nn.ModuleList()
        self.layers.append(
            HyperGCNConv(
                in_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
            )
        )
        self.layers.append(
            HyperGCNConv(
                hid_channels, num_classes, use_mediator, use_bn=use_bn, is_last=True
            )
        )
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.embedding_dict['emb']
        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    hg, X, self.with_mediator
                )
            for layer in self.layers:
                X = layer(X, hg, self.cached_g)
        else:
            for layer in self.layers:
                X = layer(X, hg)
        return X

from dhg.nn import HGNNPConv


class HGNNP(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,nums:int,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'emb': nn.Parameter(initializer(torch.empty(nums,
                                                        in_channels)))
        })

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X=self.embedding_dict['emb']
        for layer in self.layers:
            X = layer(X, hg)
        return X


class DiGCN(torch.nn.Module):
    def __init__(self,nums,in_ch, layer, dropout=0.5):
        super(DiGCN, self).__init__()
        self.dropout=dropout
        self.layers = eval(layer)
        initializer = nn.init.xavier_uniform_

        # self.embedding_dict = nn.ParameterDict({
        #     'emb': nn.Parameter(initializer(torch.empty(nums,
        #                                                 in_ch)))
        # })

        self.conv1 = DIGCNConv(in_ch, self.layers[0])
        self.conv2 = DIGCNConv(self.layers[0], self.layers[1])
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, edge_index, edge_weight,x=[]):
        if x==[]:
            x=self.embedding_dict['emb']

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x




