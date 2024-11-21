import pickle
import os
import scipy.sparse as sp
from utils import *


data_path = 'D:\\DL_project\\bearning_fault\\dataset\\JNU\\dataset.cpkl'

# load data
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# print(data[0].shape) 3660, 3072
# print(data[1])

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


#  Construct A

def Eu_dis(x):
    assert isinstance(x, np.ndarray)
    aa = np.sum(x*x, axis=1)
    ab = x @ x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def construct_A_with_KNN_from_distance(dis_mat, k):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    A = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()

        if not np.any(nearest_idx[:k] == center_idx): # 前k个都不是center_idx
            nearest_idx[k - 1] = center_idx
        
        for node_idx in nearest_idx[:k]:
            A[node_idx, center_idx] = 1.0

    return A

dis_mat = Eu_dis(data[0])
# print(dis_mat.shape) # 3660， 3660

A = construct_A_with_KNN_from_distance(dis_mat, 3)

A = sp.csr_matrix(A)
adj_orig = A
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
pos_weight = float((adj_orig.shape[0] * adj_orig.shape[0] - adj_orig.sum()) / adj_orig.sum()) # 负样本的比重
norm = adj_orig.shape[0] * adj_orig.shape[0] / float((adj_orig.shape[0] * adj_orig.shape[0] - adj_orig.sum()) * 2)
# adj_orig.diagnoal()[np.newaxis, :] -> 1, 3660
adj_norm = process_graph(adj_orig)

# label
labels = torch.LongTensor(data[1])
train_prop = .5
valid_prop = .25
indices = []

for i in range(labels.max()+1):
    index = torch.where(labels == i)[0].view(-1)
    index = index[torch.randperm(index.size(0))]
    indices.append(index)

percls_trn = int(train_prop / (labels.max()+1)*len(labels))
val_lb = int(valid_prop*len(labels))

train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
rest_idx = torch.cat([i[percls_trn:] for i in indices], dim=0)
val_idx = rest_idx[:val_lb]
test_idx = rest_idx[val_lb:]


total_data_dict = {
    'features': data[0],
    'adj_orig': adj_orig,
    'adj_norm': adj_norm,
    'pos_weight': pos_weight,
    'norm': norm,
    'labels': labels,
    'train_idx': train_idx,
    'val_idx': val_idx,
   'test_idx': test_idx
}

if __name__ == '__main__':
    save_path = 'D:\\DL_project\\bearning_fault\\DBGAN'
    data = os.path.join(save_path, 'data.pkl')
    with open(data, 'wb') as f:
        pickle.dump(total_data_dict, f)






















