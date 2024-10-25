import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer import *


class Encoder(nn.Module):
    def __init__(self, nlayers, d, nheads, input, hidden, nclass):
        super().__init__()

        self.layers = nn.ModuleList(EncoderLayer(d, nheads, input, hidden, input) for _ in range(nlayers))
        self.fc = nn.Linear(input, nclass)
    
    def forward(self, x):
        for layer in self.layers:
            out, _ = layer(x)
        
        out = out.reshape(x.shape[0], -1)
        return F.log_softmax(self.fc(out), dim=1)


if __name__ == '__main__':
    x = torch.randn(64, 3072)
    model = Encoder(2, 512, 2, 3072, 1024, 4)
    print(model(x).shape)



        
