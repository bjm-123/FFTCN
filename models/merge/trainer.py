# %%
import gc
import os
import argparse
import sys
import torch
import numpy as np
from data.loader import Sleep_Loader
from models.merge.model import MergeModel
import datetime

# %%

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = r'D:\BJM\testdata'
    save_path = r'D:\BJM\test_outputs'
    model_name = 'FFTCN'

    raw_net_path = os.path.join(save_path, 'RawModel', '1D-CNN', 'network.pth')
    wavelet_net_path = os.path.join(save_path, 'WaveletModel', '2D-CNN', 'network.pth')

    model = MergeModel(device, save_path, raw_net_path, wavelet_net_path, name=model_name)

    ratio = [0.8, 0.1, 0.1]
    seed = 0
    size = 0
    waveshape = (30, 60)

    print('=========================================finetune=========================================')
    batch_size = 32
    seq_len = 50
    n_epoch = 50
    learn_rate = 1e-5
    lamb = 1e-2
    gamma = 0.95
    alpha = 0
    train_loader = Sleep_Loader(data_path, 'train', batch_size, seq_len, ratio=ratio, seed=seed, wave='with',
                                waveshape=waveshape, size=size)
    valid_loader = Sleep_Loader(data_path, 'valid', batch_size, seq_len, ratio=ratio, seed=seed, wave='with',
                                waveshape=waveshape, size=size)
    model.finetune(train_loader, valid_loader, n_epoch, learn_rate, lamb, gamma, alpha, True)
    model.save()

    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print(f'time cost: {time_cost}')

    test_loader = Sleep_Loader(data_path, 'test', batch_size, seq_len, ratio=ratio, seed=seed, wave='with',
                               waveshape=waveshape, size=size)
    print('best:')
    model.test(test_loader, 'best_network.pth')
    print('final:')
    model.test(test_loader)

# %%
