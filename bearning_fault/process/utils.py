import numpy as np
import pickle
import os
from scipy.signal import resample
import math
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset


def load_data(path, sample_data):
    file_name = ['ib600_2.csv', 'ib800_2.csv', 'ib1000_2.csv', 'n600_3_2.csv', 'n800_3_2.csv', 'n1000_3_2.csv',
                 'ob600_2.csv', 'ob800_2.csv', 'ob1000_2.csv', 'tb600_2.csv', 'tb800_2.csv', 'tb1000_2.csv']
    data_lst = [[] for _ in range(len(file_name))]

    for file, name in enumerate(file_name):
        data = os.path.join(path, name)
        with open(data, 'r', encoding='gb18030', errors='ignore') as f:
            for line in f:
                line = float(line.strip('\n'))
                data_lst[file].append(line)
    
    for i in range(len(data_lst)):
        data_lst[i] = data_lst[i][:sample_data]

    data = np.array(data_lst, dtype=np.float32)
    return data_lst, data


def add_noise(data, snr):
    d = np.random.randn(len(data))
    P_signal = np.sum(abs(data) ** 2)
    P_d = np.sum(abs(data) ** 2)
    P_noise = P_signal / 10 ** (snr / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = data.reshape(-1) + noise
    return noise_signal


def resample_signal(data, base_sampling_rate=5000):
    sampling_rate_lst = [base_sampling_rate, base_sampling_rate/2, base_sampling_rate/4, base_sampling_rate/8]
    resample_data = []
    if isinstance(data, np.ndarray):
        for i in range(data.shape[0]):
            for ra in sampling_rate_lst:
                re_length = int(len(data[i]) * ra / base_sampling_rate)
                re_data = resample(data[i], re_length)
                resample_data.append(re_data)

    elif isinstance(data, list):
        for i in range(len(data)):
            for ra in sampling_rate_lst:
                re_length = int(len(data[i]) * ra / base_sampling_rate)
                re_data = resample(data[i], re_length)
                resample_data.append(re_data)
    else:
        raise ValueError('the type of data is wrong!!!')
    return resample_data


def splide_window_sampling(data, window_size, overlap):
    assert isinstance(data, np.ndarray)
    data_length = int(data.shape[0])
    sample_num = math.floor((data_length - window_size) / overlap + 1)
    sequece = np.zeros((sample_num, window_size), dtype=np.float32)
    for i in range(sequece.shape[0]):
        sequece[i] = data[overlap * i : window_size + overlap * i].T
    return sequece


def normalize(data):
    assert isinstance(data, np.ndarray)
    assert len(data.shape) == 1
    data_max, data_min = data.max(axis=0), data.min(axis=0)
    normal_data = data - data_min / (data_max - data_min)
    return normal_data

def FFT(data):
    assert isinstance(data, np.ndarray)
    assert len(data.shape) == 1
    fft_data = np.abs(np.fft.fft(data) / len(data))
    return fft_data

def dataloader(data, labels):
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.25, shuffle=True, stratify=Y_train_val)
    x_train, y_train, x_test, y_test, x_val, y_val = torch.FloatTensor(X_train), torch.LongTensor(Y_train), torch.FloatTensor(X_test), torch.LongTensor(Y_test), torch.FloatTensor(X_val), torch.LongTensor(Y_val)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)
    return train_loader, test_loader, val_loader

def valid(y_hat, label):
    _, pred = torch.max(y_hat, dim=1)
    acc = (pred == label).sum().item() / len(label)
    return acc

def train(train_loader, val_loader, model, num_epoch, lr, weight_decay, patience, device, weight_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    model.to(device)
    batch_loss_train_lst = []
    batch_acc_train_lst = []
    batch_loss_val_lst = []
    batch_acc_val_lst = []
    count = 0
    for epoch in range(num_epoch):
        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            out = model(data)
            l = F.nll_loss(out, label)
            batch_loss_train_lst.append(l.item())
            acc = valid(out, label)
            batch_acc_train_lst.append(acc)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_loss = sum(batch_loss_train_lst) / len(batch_loss_train_lst)
        train_acc = sum(batch_acc_train_lst) / len(batch_acc_train_lst)

        model.eval()
        with torch.no_grad():
            for data, label in val_loader:
                data = data.to(device)
                label = label.to(device)
                out = model(data)
                l = F.nll_loss(out, label)
                acc = valid(out, label)
                batch_loss_val_lst.append(l.item())
                batch_acc_val_lst.append(acc)
            val_loss = sum(batch_loss_val_lst) / len(batch_loss_val_lst)
            val_acc = sum(batch_acc_val_lst) / len(batch_acc_val_lst)
        
        print('epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch, train_loss, train_acc, 
                                                                                                           val_loss, val_acc))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weight_path, 'best_model.pth'))
            count += 0
        else:
            count += 1
        
        if count > patience:
            break

def test(model, weight_path, test_loader, device):
    model.load_state_dict(torch.load(os.path.join(weight_path, 'best_model.pth')))
    model.eval()
    acc_lst = []
    f1_lst = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            out = model(data)
            acc = accuracy_score(label.detach().cpu().numpy(), out.detach().cpu().numpy().argmax(axis=1))
            f1 = f1_score(label.detach().cpu().numpy(), out.detach().cpu().numpy().argmax(axis=1), average='micro')
            acc_lst.append(acc)
            f1_lst.append(f1)
    
    avg_acc = sum(acc_lst) / len(acc_lst)
    avg_f1 = sum(f1_lst) / len(f1_lst)
    print('avg acc: {:.2f}, avg f1 score: {:.2f}'.format(avg_acc, avg_f1))



            
        
        



    
    








    
