import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [
            nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True)
            for l in range(L)
        ]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GINConvLayer(nn.Module):

    def __init__(self, node_dim, edge_dim, dropout):
        super(GINConvLayer, self).__init__()
        emb_dim = node_dim + edge_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, node_dim))
        self.dropout = nn.Dropout(dropout)

    def message_func(self, edges):
        h_e = self.mlp(torch.cat([edges.src["h"], edges.data["e"]], dim=1))
        return {"h_e": h_e}

    def reduce_func(self, nodes):
        h = nodes.mailbox["h_e"].sum(dim=1)
        return {"h": h}

    def forward(self, g, h, e):
        g.ndata["h"] = h
        g.edata["e"] = e
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]
        e = g.edata["e"]

        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        h = self.dropout(h)
        e = self.dropout(e)

        return h, e



class GatedGCNLayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, input_dim, dropout, batch_norm, residual=False):
        super().__init__()
        output_dim = input_dim
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        e_ij = (
            edges.data["Ce"] + edges.src["Dh"] + edges.dst["Eh"]
        )  # e_ij = Ce_ij + Dhi + Ehj
        edges.data["e"] = e_ij
        return {"Bh_j": Bh_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)
        # h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (
            torch.sum(sigma_ij, dim=1) + 1e-6
        )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        return {"h": h}

    def forward(self, g, h, e):

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.edata["e"] = e
        g.edata["Ce"] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]  # result of graph convolution
        e = g.edata["e"]  # result of graph convolution

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        h = F.relu(h)  # non-linear activation
        e = F.relu(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    def __repr__(self):
        return "{}(in_channels={}, out_channels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class GatedGCNNet(nn.Module):
    def __init__(self, in_dim, in_dim_edge, hidden_dim, out_dim, dropout, n_layers):
        super().__init__()
        self.out_dim = out_dim
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True
        self.pos_enc = False
        if self.pos_enc:
            pos_enc_dim = 100
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                GINConvLayer(
                    in_dim, in_dim_edge, dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, g, h, e, h_pos_enc=None):

        # h = self.embedding_h(h.float())
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float())
            h = h + h_pos_enc
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        # e = self.embedding_e(e.float())

        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata["h"] = h
        g.edata["e"] = e

        return h, e
