import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SELayer import SE_Layer
from layers.CBAM import *

class SE_Conv(nn.Module):
    def __init__(self, in_channel, dropout, reduction, n_class):
        super().__init__()

        self.in_channel = in_channel

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=10, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=6),
            nn.BatchNorm1d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.att = SE_Layer(64, reduction) 

        self.out_layer = nn.Sequential(
            nn.Linear(64*94, 1024),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_class)
        )
    
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = self.layer5(x) 
        x = self.att(x)
        x = x.reshape(x.shape[0], -1)
        out = F.log_softmax(self.out_layer(x), dim=1)
        return out


class CBAM_Conv(nn.Module):
    def __init__(self, in_channel, reduction, dropout=0.5, n_class=4):
        super().__init__()
        self.channel_att = ChannelAttention(64, reduction)
        self.spatial_att = SptialAttention()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=10, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=6),
            nn.BatchNorm1d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.out_layer = nn.Sequential(
            nn.Linear(64*94, 1024),
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_class)
        )
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        x = self.layer5(x) # 64, 64, 94
        x_channel = self.channel_att(x)
        f_titla = x * x_channel
        x_sptial = self.spatial_att(f_titla)
        f_titla_2 = f_titla * x_sptial
        y= f_titla_2.reshape(f_titla_2.shape[0], -1)
        out = F.log_softmax(self.out_layer(y), dim=1)
        return out
    
if __name__ == '__main__':
    x = torch.randn(64, 3072)
    model = CBAM_Conv(1, 2)
    print(model(x).shape)
