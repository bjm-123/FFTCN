# %%
import os
import argparse
import numpy as np
from tqdm import tqdm
from de import DE_PSD

# %%

if __name__ == "__main__":

    source_path = r'D:\BJM\SHHS-1'
    target_path = r'D:\BJM\SHHS-clean'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    dirs = os.listdir(source_path)
    pbar = tqdm(dirs)
    for i, dir in enumerate(pbar):
        with np.load(os.path.join(source_path, dir)) as f:
            data = f['x']
            label = f['y']
        psd = np.zeros([data.shape[0], 9, 1], dtype=float)
        de = np.zeros([data.shape[0], 9, 1], dtype=float)
        for j, d in enumerate(data):
            psd[j], de[j] = DE_PSD(d.T)
        length = len(label)

        save_dict = {
            'x': de,
            'y': label,
        }
        np.savez(os.path.join(target_path, dir), **save_dict)

# %%

source_path = r'D:\BJM\SHHS-1'
target_path = r'D:\BJM\SHHS-clean'
if not os.path.exists(target_path):
    os.makedirs(target_path)

dirs = os.listdir(source_path)
pbar = tqdm(dirs)
for i, dir in enumerate(pbar):
    with np.load(os.path.join(source_path, dir)) as f:
        save_dict = {
            # 'x':np.expand_dims(f['x'], 1),
            'x': np.transpose(f['x'], (0, 2, 1)),
            'y': f['y'],
        }
    np.savez(os.path.join(target_path, dir), **save_dict)


# %%

def cut(x, y):
    c = np.where(y != 0)[0]
    begin = np.max([0, c[0] - 60])
    end = np.min([len(y), c[-1] + 61])
    cut_x = x[begin:end]
    cut_y = y[begin:end]
    return cut_x, cut_y


source_path = r'D:\BJM\SHHS-1'
target_path = r'D:\BJM\SHHS-clean'
if not os.path.exists(target_path):
    os.makedirs(target_path)

dirs = os.listdir(source_path)
pbar = tqdm(dirs)
for i, dir in enumerate(pbar):
    with np.load(os.path.join(source_path, dir)) as f:
        cut_x, cut_y = cut(f['x'], f['y'])
        save_dict = {
            # 'x':np.expand_dims(f['x'], 1),
            'x': cut_x,
            'y': cut_y,
        }
    np.savez(os.path.join(target_path, dir), **save_dict)


# %%
def balance_resample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)
    return balance_x, balance_y


source_path = r'D:\BJM\SHHS-1'
target_path = r'D:\BJM\SHHS-clean'
if not os.path.exists(target_path):
    os.makedirs(target_path)

dirs = os.listdir(source_path)
pbar = tqdm(dirs)
for i, dir in enumerate(pbar):
    with np.load(os.path.join(source_path, dir)) as f:
        br_x, br_y = balance_resample(f['x'], f['y'])
        save_dict = {
            # 'x':np.expand_dims(f['x'], 1),
            'x': br_x.astype(np.float16),
            'y': br_y.astype(np.int8),
        }
    np.savez(os.path.join(target_path, dir), **save_dict)

# %%
