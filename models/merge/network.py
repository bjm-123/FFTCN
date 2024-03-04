# %%
import torch
import torch.nn as nn
import sys
import os
sys.path.append(r'..\..')
from models.base.tcn import TemporalConvNet
from models.base.layers import LSTM
from models.raw.network import RawFeatureNet
from models.wavelet.network import WaveFeatureNet

# %%
class MergeSleepNet(nn.Module):
    def __init__(self, params, raw_feature_net:RawFeatureNet=None, wave_feature_net:WaveFeatureNet=None, n_class=5):
        super(MergeSleepNet, self).__init__()
        
        if raw_feature_net:
            self.raw_feature_net = raw_feature_net
        else:
            self.raw_feature_net = RawFeatureNet(params['raw_feature_net'])
        self.raw_feature_net.finetune()
        if wave_feature_net:
            self.wave_feature_net = wave_feature_net
        else:
            self.wave_feature_net = WaveFeatureNet(params['wave_feature_net'])
        self.wave_feature_net.finetune()
        # self.sleep_net = LSTM(**params['lstm'])
        self.sleep_net = TemporalConvNet(**params['tcn'])
        self.classifier = nn.Linear(**params['classifier'])


    def forward(self, x1, x2):
        bt_size = x1.shape[0]
        seq_len = x1.shape[1]
        x1 = x1.contiguous().view(bt_size*seq_len, *x1.shape[2:])
        f1 = self.raw_feature_net(x1)
        f1 = f1.contiguous().view(bt_size, seq_len, *f1.shape[1:])

        x2 = x2.contiguous().view(bt_size*seq_len, *x2.shape[2:])
        f2 = self.wave_feature_net(x2)
        f2 = f2.contiguous().view(bt_size, seq_len, *f2.shape[1:])

        c = torch.cat([f1, f2], -1)
        s = self.sleep_net(c)
        s = s.contiguous().view(bt_size*seq_len, *s.shape[2:])
        y = self.classifier(s)
        return y

# %%
