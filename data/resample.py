# %%
import numpy as np
from collections import Counter

def copy_resample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    counter = Counter(y)
    class_labels = counter.keys()
    n_max_classes = max(counter.values())

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


def offset_resample(x, y, offset):
    counter = Counter(y)
    class_labels = counter.keys()
    n_max_classes = max(counter.values())

    balance_x = [x]
    balance_y = [y]
    hr = np.array([0, len(y)-1])
    xt = np.reshape(x, -1)
    for c in class_labels:
        idx = np.where(y == c)[0]
        idx = np.setdiff1d(idx, hr)
        if len(idx) != 0:
            idx_re = np.random.choice(idx, n_max_classes, replace=True)
            idx_re = idx_re*3000 + np.random.randint(-offset, offset, len(idx_re))
            x_re = np.array([np.expand_dims(xt[i:i+3000], 0) for i in idx_re])
            y_re = np.full(len(idx_re), c, dtype=np.int8)
            balance_x.append(x_re)
            balance_y.append(y_re)
    
    balance_x = np.concatenate(balance_x)
    balance_y = np.concatenate(balance_y)
    return balance_x, balance_y
# %%
