import numpy as np
import scipy.sparse as sp
import os
from utils import *
import pickle

path = 'Demo_GCN/data/cora'
save_path = 'Demo_GCN/data/cora'

idx_features_labels = np.genfromtxt(os.path.join(path, 'cora.content'), dtype=np.dtype(str))

features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
# features -> 2708, 1433
labels = encode_onehot(idx_features_labels[:, -1])
# labels -> 2708, 7
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
edges_unordered = np.genfromtxt(os.path.join(path, 'cora.cites'), dtype=np.int32)

adj = build_graph(idx, edges_unordered, labels)
# adj -> 2708, 2708
# print(isinstance(adj, sp.coo_matrix))


adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj_orig.eliminate_zeros()
adj_train = mask_test_edges(adj)
adj = adj_train
adj_dense = adj.toarray()

adj_norm = process_graph(adj)
    
num_nodes = adj.shape[0]
features_dense = features.tocoo().toarray()

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]

features_nonzero = features[1].shape[0]

pos_weight = float((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()) # 负样本的比重
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

feas = {
        'adj_orig': adj_orig,
        'adj': adj,
        'adj_dense': adj_dense,
        'adj_norm': adj_norm,
        'num_nodes': num_nodes,
        'features': features,
        'features_dense': features_dense,
        'num_features': num_features,
        'features_nonzero': features_nonzero,
        'pos_weight': pos_weight,
        'norm': norm,
        'adj_label': adj_label,
        }

process_data = os.path.join(save_path, 'data.pkl')

with open(process_data, 'wb') as f:
    pickle.dump(feas, f)
f.close()


   












    













   

