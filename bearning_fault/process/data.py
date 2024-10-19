import pickle
from process.utils import *
import os

data_path = 'dataset/JNU'
save_path = 'dataset/JNU'
noise = None
window_size = 1024
overlap = 1024


data_lst, data = load_data(data_path, sample_data=500500)

if noise is not None:
    nosie_data = np.zeros_like(data, dtype=np.float32)
    for i in range(nosie_data.shape[0]):
        nosie_data[i] = add_noise(data[i], snr=1)
else:
    noise_data = data
resample_data = resample_signal(noise_data)

win_data_lst = []
for i in range(len(resample_data)):
    win_data = splide_window_sampling(resample_data[i], window_size, overlap)
    win_data_lst.append(win_data)

normalize_data_lst = []
for i in range(len(win_data_lst)):
    norm_data = np.zeros_like(win_data_lst[i])
    for j in range(norm_data.shape[0]):
        norm_data[j] = normalize(win_data_lst[i][j])
    normalize_data_lst.append(norm_data)    

fft_data_lst = []
for i in range(len(normalize_data_lst)):
    fft_data = np.zeros_like(normalize_data_lst[i])
    for j in range(fft_data.shape[0]):
        fft_data[j] = FFT(normalize_data_lst[i][j])
    fft_data_lst.append(fft_data)

# Group
ib_data = fft_data_lst[0:12]
n_data = fft_data_lst[12:24]
ob_data = fft_data_lst[24:36]
tb_data = fft_data_lst[36:48]

# Ib Fault Diagnosis Sample
ib_data_rate_1 = np.hstack([ib_data[0], ib_data[4], ib_data[8]])
ib_data_rate_2 = np.hstack([ib_data[1], ib_data[5], ib_data[9]])
ib_data_rate_3 = np.hstack([ib_data[2], ib_data[6], ib_data[10]])
ib_data_rate_4 = np.hstack([ib_data[3], ib_data[7], ib_data[11]])
ib_sample = np.vstack([ib_data_rate_1, ib_data_rate_2, ib_data_rate_3, ib_data_rate_4])
print(f'ib_sample: {ib_sample.shape}')

# n Fault Diagnosis Sample
n_data_rate_1 = np.hstack([n_data[0], n_data[4], n_data[8]])
n_data_rate_2 = np.hstack([n_data[1], n_data[5], n_data[9]])
n_data_rate_3 = np.hstack([n_data[2], n_data[6], n_data[10]])
n_data_rate_4 = np.hstack([n_data[3], n_data[7], n_data[11]])
n_sample = np.vstack([n_data_rate_1, n_data_rate_2, n_data_rate_3, n_data_rate_4])
print('n_sample: {}'.format(n_sample.shape))

# ob Fault Diagnosis Sample
ob_data_rate_1 = np.hstack([ob_data[0], ob_data[4], ob_data[8]])
ob_data_rate_2 = np.hstack([ob_data[1], ob_data[5], ob_data[9]])
ob_data_rate_3 = np.hstack([ob_data[2], ob_data[6], ob_data[10]])
ob_data_rate_4 = np.hstack([ob_data[3], ob_data[7], ob_data[11]])
ob_sample = np.vstack([ob_data_rate_1, ob_data_rate_2, ob_data_rate_3, ob_data_rate_4])
print('ob_sample: {}'.format(ob_sample.shape))

# tb Fault Diagnosis Sample
tb_data_rate_1 = np.hstack([tb_data[0], tb_data[4], tb_data[8]])
tb_data_rate_2 = np.hstack([tb_data[1], tb_data[5], tb_data[9]])
tb_data_rate_3 = np.hstack([tb_data[2], tb_data[6], tb_data[10]])
tb_data_rate_4 = np.hstack([tb_data[3], tb_data[7], tb_data[11]])
tb_sample = np.vstack([tb_data_rate_1, tb_data_rate_2, tb_data_rate_3, tb_data_rate_4])
print('tb_sample: {}'.format(tb_sample.shape))
# total sample
sample_data = np.vstack([ib_sample, n_sample, ob_sample, tb_sample])
labels = np.full(sample_data.shape[0], -1, dtype=np.int32)
labels[:915] = 0
labels[915:1830] = 1
labels[1830:2745] = 2
labels[2745:3660] = 3


dataset = os.path.join(save_path, 'dataset.cpkl')
with open(dataset, 'wb') as f:
    pickle.dump((sample_data, labels), f, protocol=2)
f.close()


