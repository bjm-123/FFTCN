# %%
import torch
import torch.nn as nn
import sys
sys.path.append(r'..\..')
from models.base.tcn import TemporalConvNet
from models.base.layers import LSTM
# %%
class Conv1dLayer(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(**params),
            nn.BatchNorm1d(params['out_channels']),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class RawFeatureNet(nn.Module):
    def __init__(self, params, n_class=5):
        super(RawFeatureNet, self).__init__()
        cnn_params = params['cnn']
        model = []
        model.append(Conv1dLayer(cnn_params[0]))
        model.append(nn.MaxPool1d(**cnn_params[1])) # kernel_size stride padding
        model.append(nn.Dropout(**cnn_params[2]))
        model.append(Conv1dLayer(cnn_params[3]))
        model.append(Conv1dLayer(cnn_params[4]))
        model.append(Conv1dLayer(cnn_params[5])) 
        model.append(nn.MaxPool1d(**cnn_params[6]))
        model.append(nn.Dropout(**cnn_params[7]))

        self.model = nn.Sequential(*model)
        classifier = []
        classifier.append(Conv1dLayer(params['classifier'][0]))
        classifier.append(nn.AdaptiveAvgPool1d(**params['classifier'][1]))

        self.classifier = nn.Sequential(*classifier)
        self.classifier_flag = True


    def forward(self, x):
        y = self.model(x)
        if self.classifier_flag:
            y = self.classifier(y)
        y = torch.flatten(y, -2, -1)
        return y

    
    def pretrain(self):
        self.classifier_flag = True

    
    def finetune(self):
        self.classifier_flag = False


class RawSleepNet(nn.Module):
    def __init__(self, params, feature_net:RawFeatureNet=None, n_class=5):
        super(RawSleepNet, self).__init__()
        
        if feature_net:
            self.feature_net = feature_net
        else:
            self.feature_net = RawFeatureNet(params['feature_net'])
        # self.sleep_net = LSTM(**params['lstm'])
        self.sleep_net = TemporalConvNet(**params['tcn'])
        self.classifier = nn.Linear(**params['classifier'])


    def forward(self, x):
        bt_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.contiguous().view(bt_size*seq_len, *x.shape[2:])
        f = self.feature_net(x)
        f = f.contiguous().view(bt_size, seq_len, *f.shape[1:])
        s = self.sleep_net(f)
        s = s.contiguous().view(bt_size*seq_len, *s.shape[2:])
        y = self.classifier(s)
        return y

# %%
