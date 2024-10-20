import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SELayer import SELayer

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

        self.att = SELayer(64, reduction) 

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
        x = self.att(x)
        x = x.reshape(x.shape[0], -1)
        out = F.log_softmax(self.out_layer(x), dim=1)
        return out
    
if __name__ == '__main__':
    x = torch.randn(64, 3072)
    model = SE_Conv(1, 0.5, 4)
    print(model(x).shape)
