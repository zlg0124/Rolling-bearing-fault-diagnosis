import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pickle
from DBGAN.process import *


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

    



