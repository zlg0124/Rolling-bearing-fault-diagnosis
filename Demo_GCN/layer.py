import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pickle
from utils import *


class GraphConvoluationSparse(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., bias=None):
        super().__init__()
        self.w  = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = bias
        if self.bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.dropout = dropout
        self.relu = nn.ReLU(inplace=0.2)
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
        x = torch.sparse.mm(x, self.w)
        output = torch.sparse.mm(adj, x)
        if self.bias is not None:
            output = output + self.bias
        return output
    
    def dropout_sparse(self, x, keep_prob):
        mask = torch.bernoulli(torch.full_like(x.values(), keep_prob)).bool()
        x = torch.sparse_coo_tensor(x.indices(), x.values()*mask. x.size())
        return x / keep_prob


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        
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
        x = torch.mm(x, self.w)
        output = torch.sparse.mm(adj, x)
        if self.bias is not None:
            output = output + self.bias
        
        return self.relu(output)
    

class InnerProductDecoder(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout(x)
        x = torch.mm(x, x.T).reshape(-1)
        return F.sigmoid(x)



if __name__ == '__main__':
    with open('Demo_GCN/data/cora/data.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    f.close()
    features = data_dict['features']
    features = torch.sparse_coo_tensor(features[0].T, features[1], features[2])
    adj = sparse_mx_to_torch_sparse_tensor(data_dict['adj'])
    layer = GraphConvoluationSparse(features.shape[1], 64)
    print(layer(features, adj).shape)


    



