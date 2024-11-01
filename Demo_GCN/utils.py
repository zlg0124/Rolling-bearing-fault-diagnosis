import numpy as np
import scipy.sparse as sp
import torch

# Perform one-hot encoding on labels
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c : np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot 

# build Graph
def build_graph(idx, edges_unordered, labels):
    idx_map = {j : i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), 
                        shape=(labels.shape[0], labels.shape[0]))
    return adj

def process_graph(mx):
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return sparse_to_tuple(mx)


def sparse_to_tuple(mx):
    if not isinstance(mx, sp.coo_matrix):
        mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape


def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return np.any(rows_close)



def mask_test_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape) # 移除对角线元素
    adj.eliminate_zeros() # 移除所有的0元素
    
    assert np.diag(adj.todense()).sum() == 0 # 检查对角线元素是否都是0
    # 创建上三角矩阵，使得每条边保留一次
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    # 取edge的10%作为test
    # 取edge的20%作为val
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_index = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_index)

    val_edge_idx = all_edge_index[:num_val]
    test_edge_idx = all_edge_index[num_val:(num_test+num_val)]

    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]

    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # test_edges_false存储不存在的edges

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])

        if idx_i == idx_j:
            continue

        if ismember([idx_i, idx_j], edges_all):
            continue

        if test_edges_false:
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        
        test_edges_false.append([idx_i, idx_j])
    
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])

        if idx_i == idx_j:
            continue

        if ismember([idx_i, idx_j], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    
    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)

    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    
    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)

    adj_train = adj_train + adj_train.T

    return adj_train

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)



if __name__ == '__main__':
    a = np.random.randint(0, 10)
    b = np.random.randint(0, 10)
    c = np.array([[1, 3], [5, 6]])
    print(ismember([a, b], c))

    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = x[np.newaxis, :]
    print(y.shape) # 1, 2, 3








