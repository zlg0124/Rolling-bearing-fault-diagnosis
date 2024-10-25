import torch
import torch.nn as nn
import math

class ScaleDotAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, Q, K, V):
        # Q -> batch_size, qkv, num_hiddens
        # K -> batch_size, qkv, num_hiddens
        # V - > batch_size, qkv, dv
        scores = torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Q.shape[2])
        out = torch.bmm(self.softmax(scores), V)
        return out, self.softmax(scores)


class MultiHeadAttention(nn.Module):
    def __init__(self, nheads, input_dim, d):
        super().__init__()
        self.nheads = nheads
        self.att = ScaleDotAttention()
        self.Wq = nn.Linear(input_dim, nheads*d)
        self.Wk = nn.Linear(input_dim, nheads*d)
        self.Wv = nn.Linear(input_dim, nheads*d)
        self.linear = nn.Linear(nheads*d, input_dim)
        self.layernorm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)
        Q = self.transpose_qkv(self.Wq(x), self.nheads)
        K = self.transpose_qkv(self.Wk(x), self.nheads)
        V = self.transpose_qkv(self.Wv(x), self.nheads)
        att_output, weights = self.att(Q, K, V)
        # print(att_output.shape)
        att_output = self.inv_transpose_qkv(att_output, self.nheads) # output -> batch_size*nheads, qkv, hidden
        output = self.linear(att_output)
        residual = x + output
        # print(residual.shape)
        return self.layernorm(residual), weights

    def transpose_qkv(self, x, nheads):
        # x -> batch_size, qkv, heads*hidden -> batch_size, qkv, heads, hidden
        x = x.reshape(x.shape[0], x.shape[1], nheads, -1)
        x = x.permute(0, 2, 1, 3) # x-> batch_size, nheads, qkv, hidden
        x = x.reshape(-1, x.shape[2], x.shape[3]) # batch_size*nheads, qkv, hidden
        return x
    
    def inv_transpose_qkv(self, x, nheads):
        # x -> batch_size*nheads, qkv, hidden -> batch_size, nheads, qkv, hidden
        x = x.reshape(-1, nheads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x


class PositionWiseFFN(nn.Module):
    def __init__(self, input, hidden, output):
        super().__init__()
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output)
        self.layernorm = nn.LayerNorm(output)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.layer2(self.relu(self.layer1(x)))
        out = x + residual
        return self.layernorm(out)

class EncoderLayer(nn.Module):
    def __init__(self, d, nheads, input, hidden, output):
        super().__init__()
        self.att = MultiHeadAttention(nheads, input, d)
        self.pos_ffn = PositionWiseFFN(input, hidden, output)
    
    def forward(self, x):
        att_out, weights = self.att(x) 
        out = self.pos_ffn(att_out)
        return out, weights

if __name__ == '__main__':
    x = torch.randn(64, 3072)
    layer = EncoderLayer(512, 2, 3072, 1024, 3072)
    x, _ = layer(x)
    print(x.shape)
    