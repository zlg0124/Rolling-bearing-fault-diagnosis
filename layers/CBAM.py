import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channel, in_channel//reduction, 1, bias=False)
        self.fc2 = nn.Conv1d(in_channel//reduction, in_channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SptialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv_7x7 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv_7x7(torch.cat([avg_out, max_out], dim=1))
        return F.sigmoid(out)
    
    
