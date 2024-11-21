import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *


"""
原始代码中, hidden1=64, hidden2=128, hidden3=64
"""
class Reconstruction(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.construct = InnerProductDecoder(in_dim, dropout)
    
    def forward(self, x):
        return self.construct(x)


class Encoder(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, dropout=0.2, bias=True):
        super().__init__()

        self.layer1 = GraphConvolution(in_features, hidden1, dropout, bias)
        self.layer2 = GraphConvolution(hidden1, hidden2, dropout, bias)
        self.recons = Reconstruction(hidden2)
    

    def forward(self, x, adj):
        # x -> 2708, 1433
        hidden = self.layer1(x, adj)
        self.embeddings = self.layer2(hidden, adj) # self.embeddings -> 2708, 128
        reconstructions = self.recons(self.embeddings)
        return self.embeddings, reconstructions
    

class Generator_z2g(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, dropout=0.2, bias=True):
        super().__init__()
        # in_features: 特征矩阵的维度
        self.d = dropout
        self.layer1 = GraphConvolution(hidden2, hidden1, dropout, bias)
        self.layer2 = GraphConvolution(hidden1, in_features, dropout, bias)

    def forward(self, x, adj):
        hidden = self.layer1(x, adj)
        hidden = F.dropout(hidden, p=self.d)
        embeddings = self.layer2(hidden, adj)
        return embeddings

  
class D_graph(nn.Module):
    def __init__(self, in_features, hidden2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 512)
        self.linear2 = nn.Linear(512, hidden2)
        self.linear3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        hidden = self.relu(self.linear1(x))
        hidden_1 = self.relu(self.linear2(hidden))
        out = self.linear3(hidden_1)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden1, hidden3):
        super().__init__()
        # 原始的features经过Encoder -> 2708, 1433 -> 2708, hidden2
        self.linear1 = nn.Linear(in_dim, hidden3)
        self.linear2 = nn.Linear(hidden3, hidden1)
        self.linear3 = nn.Linear(hidden1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        hidden_1 = self.relu(self.linear1(x))
        hidden_2 = self.relu(self.linear2(hidden_1))
        out = self.linear3(hidden_2)
        return out


if __name__ == '__main__':
    model = Encoder(1433, 64, 128)
    x = torch.randn(2708, 1433)
    adj = torch.randn(2708, 2708)
    out = model(x, adj)
    print(out[0])
    print(model.embeddings)


  


    





    
