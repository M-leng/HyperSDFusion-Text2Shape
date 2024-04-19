import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
from torch_geometric.data import Data, Batch
import geoopt
import geoopt_layers
import torch.nn.functional as F
from geoopt_layers.poincare.graph.graph_conv import HyperbolicGraphConv

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]

class TextGraph(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            n_embed=None
    ):
        super().__init__()
        self.n_embed = n_embed
        self.token_emb = nn.Embedding(num_tokens, n_embed)
        self.pos_emb = AbsolutePositionalEmbedding(n_embed, 77)
        self.emb_dropout = nn.Dropout(0.01)
        self.project_emb = nn.Sequential(nn.Linear(n_embed, n_embed),
                                         nn.SiLU(),
                                         nn.Linear(n_embed, n_embed),
                                         nn.SiLU(),
                                         nn.Linear(n_embed, n_embed),
                                         nn.SiLU(),
                                         nn.Linear(n_embed, n_embed),
                                         nn.SiLU(),
                                         nn.Linear(n_embed, n_embed))

        self.max_seq_len = max_seq_len
        self.init_()
        self.manifold = geoopt.PoincareBall(learnable=True)
        self.gcn_layer = nn.ModuleList()
        self.gcn_layer.append(
            HyperbolicGraphConv(n_embed, n_embed, aggr="sum", bias=True,
                                learn_origin=True, ball=geoopt.PoincareBallExact(learnable=True))
        )
        self.gcn_layer.append(
            HyperbolicGraphConv(n_embed, n_embed, aggr="sum", bias=True,
                                learn_origin=True, ball=geoopt.PoincareBallExact(learnable=True))
        )
        self.gcn_layer.append(
            HyperbolicGraphConv(n_embed, n_embed, aggr="sum", bias=True,
                                learn_origin=True, ball=geoopt.PoincareBallExact(learnable=True))
        )
        self.gcn_layer.append(
            HyperbolicGraphConv(n_embed, n_embed, aggr="sum", bias=True,
                                learn_origin=True, ball=geoopt.PoincareBallExact(learnable=True))
        )

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def hyperbolic_ReLU(self, hyperbolic_input, manifold):
        euclidean_input = manifold.logmap0(hyperbolic_input)
        euclidean_output = F.leaky_relu(euclidean_input)
        hyperbolic_output = manifold.expmap0(euclidean_output)
        return hyperbolic_output

    def forward(self,tokens, edge):
        batch = tokens.shape[0]
        x = self.token_emb(tokens)
        x += self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        data_list = []
        for i in range(batch):
            edge_elem = sp.coo_matrix(edge[i].cpu())
            edge_index = torch.LongTensor(np.vstack((edge_elem.row, edge_elem.col))).to(x.device)
            graph_item = Data(x=x[i], edge_index=edge_index)
            data_list.append(graph_item)
        mini_batch = Batch.from_data_list(data_list)
        graph_node = mini_batch.x
        graph_edge = mini_batch.edge_index
        graph_node = self.manifold.expmap0(graph_node, dim=-1)
        for layer in self.gcn_layer:
            x = layer(graph_node, graph_edge)
            x = self.hyperbolic_ReLU(x, self.manifold)
        x = self.manifold.logmap0(x)
        x = x.view(batch, -1, self.n_embed)
        return x