import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channel, in_channel//reduction)
        self.fc2 = nn.Linear(in_channel//reduction, in_channel)
    
    def forward(self, x):
        assert len(x.size()) == 3
        b, c, _ = x.size()
        y = self.avg_pool(x) # y -> b, c, 1
        y = y.reshape(b, -1)
        y = self.fc1(y) # b, c//reduction
        y = self.fc2(y) # b, c
        y = F.sigmoid(y)
        y = y.reshape(b, c, 1)
        return x * y

if __name__ == '__main__':
    x = torch.randn(64, 64, 94)
    layer = SELayer(64, 2)
    print(layer(x).shape)




