import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def conv1d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    input_length = input.size(2)
    filter_length = weight.size(2)
    out_length = (input_length + stride[0] - 1) // stride[0]
    padding_length = max(0, (out_length - 1) * stride[0] + (filter_length - 1) * dilation[0] + 1 - input_length)
    
    length_odd = (padding_length % 2 != 0)
    if length_odd:
        input = F.pad(input, [0, int(length_odd)])
        
    return F.conv1d(input, weight, bias, stride, padding=padding_length//2 , dilation=dilation, groups=groups)

class Conv1dTCN(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causal=True):
        self.causal = causal
        if causal:
            # double the output and chomp it
            padding = (kernel_size-1) * dilation
        else: 
            # set padding for zero for non-causal to padd in forward
            padding = 0
        super(Conv1dTCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
    def forward(self, input):
        if self.causal:
            x = F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = x[:, :, :-self.padding[0]].contiguous()
            return x
        else:
            return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, causal=True):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(Conv1dTCN(n_inputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation, causal=causal))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(Conv1dTCN(n_outputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation, causal=causal))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channle, num_channels, kernel_size=2, dropout=0.2, causal=True, channle_last=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.channle_last = channle_last
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_channle if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     dropout=dropout, causal=causal)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.channle_last:
            x = x.permute(0, 2, 1)
        y = self.network(x)
        if self.channle_last:
            y = y.permute(0, 2, 1)
        return y