import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pickle
from utils import *


class GraphConvoluationSparse(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., bias=True):
        super().__init__()
        self.w  = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = bias
        if self.bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('b', None)
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.reset_params()
    
    def reset_params(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        # x -> features稀疏矩阵
        # adj稀疏矩阵
        if self.dropout > 0.:
            x = self.dropout_sparse(x, 1-self.dropout)
        hidden = torch.sparse.mm(x, self.w)
        output = torch.sparse.mm(adj, hidden)
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def dropout_sparse(self, x, keep_prob):
        mask = torch.bernoulli(torch.full_like(x.values(), keep_prob)).bool()
        out = torch.sparse_coo_tensor(x.indices(), x.values()*mask, x.size())
        return out / keep_prob


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.bias = bias
        self.relu = nn.ReLU()
        
        self.w = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        if self.bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_params()
    
    def reset_params(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout)
        hidden = torch.mm(x, self.w)
        output = torch.sparse.mm(adj, hidden)
        if self.bias is not None:
            output = output + self.bias
        
        return self.relu(output)
    

class InnerProductDecoder(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        hidden = self.dropout(x)
        out = torch.mm(hidden, hidden.T).reshape(-1)
        return F.sigmoid(out)



if __name__ == '__main__':
    with open('Demo_GCN/data/cora/data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    f.close()
    features = data_dict['features']
    features = torch.sparse_coo_tensor(features[0].T, features[1], features[2])
    adj = sparse_mx_to_torch_sparse_tensor(data_dict['adj'])
    layer = GraphConvoluationSparse(features.shape[1], 64)
    print(layer(features, adj).shape)


    



