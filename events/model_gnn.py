import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math


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


class MDNReadout(nn.Module):
    def __init__(self, input_dim, output_dim, n_components):
        super(MDNReadout, self).__init__()
        self.coefficient_layer = nn.Sequential(nn.Linear(input_dim*2, input_dim), nn.ReLU(), nn.Linear(input_dim, n_components, bias=False))
        self.logit_layer = nn.Sequential(nn.Linear(input_dim*2, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim*n_components))
        self.n_components = n_components
        self.output_dim = output_dim

    def forward(self, graphs):
        u = []
        v = []
        edge_to_graph = []
        for i, g in enumerate(graphs):
            node_embs = g.ndata["h"]
            u.append(node_embs[g.edges()[0]])
            v.append(node_embs[g.edges()[1]])
            edge_to_graph += [i] * len(g.edges()[0])
        u = torch.cat(u)
        v = torch.cat(v)
        edge_to_graph = torch.tensor(edge_to_graph).to(u.device)

        x = torch.cat([u, v], dim=1)
        mixture_coefficients = self.coefficient_layer(x).reshape(x.shape[0], self.n_components)
        logits = self.logit_layer(x).reshape(x.shape[0], self.n_components, -1)

        for i, g in enumerate(graphs):
            mask = edge_to_graph == i
            g.edata["mixture_coefficients"] = mixture_coefficients[mask]
            g.edata["logits"] = logits[mask]
            g.edata["probs"] = torch.softmax(logits[mask], dim=2)

        return graphs


class NeighborhoodTransformer(nn.Module):

    def __init__(self, node_dim, edge_dim, dropout=0.2, output_layer=False):
        super(NeighborhoodTransformer, self).__init__()
        assert node_dim == edge_dim
        transformer_layer = nn.TransformerEncoderLayer(node_dim, 8, node_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, 3)
        self.readout_embedding = nn.Embedding(1, node_dim)

    def message_func(self, edges):
        h_e = edges.src["h"] + edges.data["e"]
        return {"h_e": h_e}

    def reduce_func(self, nodes):
        messages = nodes.mailbox["h_e"]
        readout_messages = nodes.data["h"] + self.readout_embedding(torch.tensor(0).to(messages.device))
        x = torch.cat([readout_messages.unsqueeze(1), messages], dim=1) # first message is the readout message
        h = self.transformer(nodes.mailbox["h_e"].transpose(0, 1)).transpose(0,1)[:, 0, :]
        return {"h": h}

    def forward(self, g, h, e):
        g.ndata["h"] = h
        g.edata["e"] = e
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]

        return h




class GINConvLayer(nn.Module):

    def __init__(self, node_dim, edge_dim, dropout,
                 output_layer=False):
        super(GINConvLayer, self).__init__()
        emb_dim = node_dim + edge_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, node_dim))
        self.dropout = nn.Dropout(dropout)
        self.output_layer = output_layer

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

        if not self.output_layer:
            h = F.relu(h)  # non-linear activation
            e = F.relu(e)  # non-linear activation

        h = self.dropout(h)
        e = self.dropout(e)

        return h, e



class GatedGCNLayer(nn.Module):
    """
        Param: []
    """

    def __init__(self, node_dim, edge_dim, dropout):
        super().__init__()
        assert node_dim == edge_dim
        input_dim = node_dim

        output_dim = input_dim
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = True
        self.residual = True

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


class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout, n_layers,
                 layer_type):
        super().__init__()
        self.batch_norm = True
        self.residual = True
        self.edge_feat = True
        self.pos_enc = False

        if layer_type == "GIN":
            layer_cls = GINConvLayer
        elif layer_type == "GGCNN":
            layer_cls = GatedGCNLayer
        elif layer_type == "Transformer":
            layer_cls = NeighborhoodTransformer
        else:
            raise ValueError(layer_type)

        self.layers = nn.ModuleList(
            [
                layer_cls(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    dropout=dropout
                )
                for _ in range(n_layers-1)])
        self.layers.append(
            layer_cls(
                node_dim=node_dim,
                edge_dim=edge_dim,
                dropout=dropout,
                output_layer=True
            )
        )

    def forward(self, g, h, e):
        for conv in self.layers:
            h = conv(g, h, e)
        g.ndata["h"] = h

        return h

