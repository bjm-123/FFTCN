# %%
import sys

sys.path.append(r'..\..')
import torch
import torch.nn as nn

from models.base.tcn import TemporalConvNet
from models.base.layers import LSTM


# %%
class Conv2dLayer(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(**params),
            nn.BatchNorm2d(params['out_channels']),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class WaveFeatureNet(nn.Module):
    def __init__(self, params, n_class=5):
        super(WaveFeatureNet, self).__init__()
        model = []

        # vgg
        vgg_params = params['vgg']
        model.append(Conv2dLayer(vgg_params[0]))
        model.append(Conv2dLayer(vgg_params[1]))
        model.append(nn.MaxPool2d(**vgg_params[2]))  # kernel_size stride padding
        model.append(Conv2dLayer(vgg_params[3]))
        model.append(Conv2dLayer(vgg_params[4]))
        model.append(nn.MaxPool2d(**vgg_params[5]))
        model.append(Conv2dLayer(vgg_params[6]))
        model.append(Conv2dLayer(vgg_params[7]))
        model.append(nn.MaxPool2d(**vgg_params[8]))
        model.append(Conv2dLayer(vgg_params[9]))
        model.append(Conv2dLayer(vgg_params[10]))
        model.append(nn.MaxPool2d(**vgg_params[11]))
        ##
        # for i in range(3):
        #     model.append(Conv2dLayer(vgg_params[12]))
        # model.append(nn.MaxPool2d(**vgg_params[13]))

        # ftcn
        # ftcn_param = params['ftcn']
        # for p in ftcn_param:
        #     model.append(TemporalConvNet(**p))
        # model.append(nn.AdaptiveAvgPool1d(1))

        self.model = nn.Sequential(*model)

        classifier = []
        classifier.append(Conv2dLayer(params['classifier'][0]))
        classifier.append(nn.AdaptiveAvgPool2d(**params['classifier'][1]))

        self.classifier = nn.Sequential(*classifier)
        self.classifier_flag = True

    def forward(self, x):
        # x = torch.squeeze(x)
        y = self.model(x)
        if self.classifier_flag:
            y = self.classifier(y)
        y = torch.flatten(y, 1)
        return y

    def pretrain(self):
        self.classifier_flag = True

    def finetune(self):
        self.classifier_flag = False


class WaveSleepNet(nn.Module):
    def __init__(self, params, feature_net: WaveFeatureNet = None, n_class=5):
        super(WaveSleepNet, self).__init__()

        if feature_net:
            self.feature_net = feature_net
        else:
            self.feature_net = WaveFeatureNet(params['feature_net'])
            self.feature_net.finetune()
        # self.sleep_net = LSTM(**params['lstm'])
        self.sleep_net = TemporalConvNet(**params['tcn'])
        self.classifier = nn.Linear(**params['classifier'])

    def forward(self, x):
        bt_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.contiguous().view(bt_size * seq_len, *x.shape[2:])
        f = self.feature_net(x)
        f = f.contiguous().view(bt_size, seq_len, *f.shape[1:])
        # print(f.shape)
        s = self.sleep_net(f)
        # print(s.shape)
        s = s.contiguous().view(bt_size * seq_len, *s.shape[2:])
        y = self.classifier(s)
        return y


# %%
