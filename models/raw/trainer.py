# %%
import gc
import os
import argparse
import sys
import torch
import numpy as np
from data.loader import Sleep_Loader
from models.raw.model import RawModel
import datetime

# %%

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = r'D:\BJM\testdata'
    save_path = r'D:\BJM\test_outputs'
    model_name = '1D-CNN'

    ratio = [0.8, 0.1, 0.1]
    seed = 0

    model = RawModel(device, save_path, name=model_name)
    print('=========================================pretrain=========================================')
    batch_size = 128
    n_epoch = 20
    learn_rate = 1e-5
    gamma = 0.95
    train_loader = Sleep_Loader(data_path, 'train', batch_size, ratio=ratio, seed=seed, balance=True)
    valid_loader = Sleep_Loader(data_path, 'valid', batch_size, ratio=ratio, seed=seed, balance=False)
    model.pretrain(train_loader, valid_loader, n_epoch, learn_rate, gamma)
    model.save()

    end_time = datetime.datetime.now()
    time_cost = end_time-start_time
    print(f'time cost: {time_cost}')

# %%
