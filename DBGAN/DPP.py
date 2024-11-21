from pydpp.dpp import DPP
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

def distribution(data, n_components):

    dpp = DPP(data['adj_norm'].toarray())
    dpp.compute_kernel(kernel_type='rbf', sigma=0.4)
    index = dpp.sample_k(14)

    pca = PCA(n_components)
    features_sample= data['features']
    features_sample = pca.fit_transform(features_sample)

    featuresCompress = np.array([features_sample[i] for i in index])
    #featuresCompress = np.array(feature_sample)
    kde = KernelDensity(bandwidth=0.7).fit(featuresCompress)

    # sample = kde.sample(feas['adj'].toarray().shape[0])
    return kde


if __name__ == '__main__':
    with open('Demo_GCN/data/cora/data.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()
    kde = distribution(data)
    z_real_dist = kde.sample(data['adj'].toarray().shape[0])
    print(z_real_dist.shape)
    
