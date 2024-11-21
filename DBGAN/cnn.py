
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.sparse as sp
from utils import *
from sklearn.metrics import f1_score, accuracy_score

path = 'Demo_GCN/data/cora'
save_path = 'Demo_GCN/data/cora'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'D:\\DL_project\\bearning_fault\\DBGAN\\emb_lst.pkl'
with open(data_dir, 'rb') as f:
    emb_lst = pickle.load(f)

with open('D:\\DL_project\\bearning_fault\\DBGAN\\data.pkl', 'rb') as f:
    res_data = pickle.load(f)



emb_lsts = emb_lst['emb']
for i in range(len(emb_lsts)):
    emb_lsts[i] = emb_lsts[i].to(device)

embedding = emb_lsts[-1]
# print(embedding.shape) 2708, 256

features = torch.FloatTensor(res_data['features']).to(device)
label = res_data['labels'].to(device)
train_idx = res_data['train_idx'].to(device)
val_idx = res_data['val_idx'].to(device)
test_idx = res_data['test_idx'].to(device)



def valid(y_hat, label):
    _, preds = torch.max(y_hat, dim=1)
    acc = (preds == label).sum().item() / len(label)
    return acc



class Conv(nn.Module):

    def __init__(self, in_channel, n_class):
        super().__init__()
        self.in_channel = in_channel
        self.n_class = n_class

        self.layer1 = nn.Sequential(nn.Conv1d(in_channel, 8, kernel_size=10, padding=1),
                                    nn.Dropout(0.2),
                                    nn.BatchNorm1d(8),
                                    nn.ReLU(),
                                    nn.MaxPool1d(2, 2))
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=8),
            nn.Dropout(0.2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )

        self.layer3 = nn.Sequential(
            # nn.Linear(16*417, 128),
            nn.Linear(16*762, 256),
            nn.Dropout(0.5),
        )

        self.concat_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(256, n_class),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


    def forward(self, x, embedding):
        # x = F.dropout(x, p=0.5)
        x = x.unsqueeze(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer3(x)
    
        # print(x.shape)
        x = torch.concat((x, embedding), dim=1)
        x = self.concat_layer(x)
        # print(x.shape)
        x = self.out_layer(x)
        return F.softmax(x, dim=1)
    

if __name__ == '__main__':

    model = Conv(1, label.max().item() + 1).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    best_val_loss = float('inf')
    

    for epoch in range(400):
        model.train()
        optimizer.zero_grad()
        y_hat = model(features, embedding)
        train_loss = loss(y_hat[train_idx], label[train_idx])
        train_acc = valid(y_hat[train_idx], label[train_idx])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(features, embedding)
            val_loss = loss(preds[val_idx], label[val_idx])
            val_acc = valid(preds[val_idx], label[val_idx])

            
        if (epoch+1) % 10 == 0:
            print('epoch: {}, train_loss: {:.4f}, train_acc: {:.2f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch, train_loss, train_acc, val_loss, val_acc))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    

    # Test

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        y_hat = model(features, embedding)
        acc = accuracy_score(label[val_idx].detach().cpu().numpy(), y_hat[val_idx].detach().cpu().numpy().argmax(axis=1))
        f1 = f1_score(label[val_idx].detach().cpu().numpy(), y_hat[val_idx].detach().cpu().numpy().argmax(axis=1), average='micro')

        print(f'ACC: {acc}, F1_Score: {f1}')



        
        
















    
