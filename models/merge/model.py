# %%
from models.base.utils import EarlyStopping
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
import matplotlib.pyplot as plt

from models.raw.network import RawFeatureNet
from models.wavelet.network import WaveFeatureNet
from models.merge.network import MergeSleepNet
from models.merge.parameter import Parameter
from models.base.utils import Metrics
from models.base.model import BaseModel
# %%
class MergeModel(BaseModel):
    def __init__(self, device, save_path, raw_net_path=None, wave_net_path=None, fold=None, name=None):
        BaseModel.__init__(self, device, save_path, fold, name)

        if raw_net_path:
            net = torch.load(raw_net_path)
            self.raw_feature_net = net.feature_net
        else:
            self.raw_feature_net = RawFeatureNet(Parameter.raw_feature_net, Parameter.n_class)
        if wave_net_path:
            net = torch.load(wave_net_path)
            self.wave_feature_net = net.feature_net
        else:
            self.wave_feature_net = WaveFeatureNet(Parameter.wave_feature_net, Parameter.n_class)
        self.sleep_net = MergeSleepNet(Parameter.sleep_net, self.raw_feature_net, self.wave_feature_net)


    def _train(self, net, train_loader, epoch, n_epoch, optimizer:optim.Optimizer, alpha=0):
        net.train()
        acc = 0
        loss = 0
        sys.stdout.flush()
        counter = train_loader.dataset.all_counter
        weight = torch.tensor([counter[k] for k in sorted(counter.keys())])
        weight = torch.pow(torch.sum(weight)/weight, alpha)
        weight = weight.to(self.device)
        pbar = tqdm(train_loader, desc=f'EPOCH[{epoch+1}/{n_epoch}]', ncols=100)
        for bt, (raw_inputs, wave_inputs, targets) in enumerate(pbar):
            raw_inputs = raw_inputs.to(self.device)
            wave_inputs = wave_inputs.to(self.device)
            targets = targets.to(self.device)
            if len(targets.shape) == 2:
                targets = targets.contiguous().view(targets.shape[0]*targets.shape[1])

            outputs = net(raw_inputs, wave_inputs)
            losses = F.cross_entropy(outputs, targets, weight)
            # update loss metric
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            acc_bt, loss_bt = self.evaluate(targets, outputs, losses)
            acc += acc_bt
            loss += loss_bt
            pbar.set_postfix({'acc': acc_bt, 'loss': loss_bt})
        return np.round(acc/len(train_loader)*100, 2), np.round(loss/len(train_loader), 4)


    def _valid(self, net, valid_loader):
        net.eval()
        acc = 0
        loss = 0
        for bt, (raw_inputs, wave_inputs, targets) in enumerate(valid_loader):
            with torch.no_grad():
                raw_inputs = raw_inputs.to(self.device)
                wave_inputs = wave_inputs.to(self.device)
                targets = targets.to(self.device)
                if len(targets.shape) == 2:
                    targets = targets.contiguous().view(targets.shape[0]*targets.shape[1])
                outputs = net(raw_inputs, wave_inputs)
                losses = F.cross_entropy(outputs, targets)
            acc_bt, loss_bt = self.evaluate(targets, outputs, losses)
            acc += acc_bt
            loss += loss_bt
        return np.round(acc/len(valid_loader)*100, 2), np.round(loss/len(valid_loader), 4)

    # 先使用特征网络各自的pretrain 再直接使用finetune
    def finetune(self, train_loader, valid_loader, n_epoch, learn_rate, lamb, gamma, alpha, early_stoping=False):
        f1_params = list(map(id, self.raw_feature_net.parameters()))
        f2_params = list(map(id, self.wave_feature_net.parameters()))
        s_params = filter(lambda p: id(p) not in f1_params+f2_params, self.sleep_net.parameters())
        optimizer = optim.Adam(
            params=[
                {'params': s_params}, 
                {'params': self.raw_feature_net.parameters(), 'lr': learn_rate*lamb}, 
                {'params': self.wave_feature_net.parameters(), 'lr': learn_rate*lamb},
            ],
            lr=learn_rate
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        history = {
            'train_acc':[],
            'train_loss':[],
            'valid_acc':[],
            'valid_loss':[]
        }
        self.sleep_net.to(self.device)
        self.raw_feature_net.finetune()
        self.wave_feature_net.finetune()
        if early_stoping:
            es = EarlyStopping(self.save_path)
        for epoch in range(n_epoch):
            train_acc, train_loss = self._train(self.sleep_net, train_loader, epoch, n_epoch, optimizer, alpha)
            valid_acc, valid_loss = self._valid(self.sleep_net, valid_loader)
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['valid_acc'].append(valid_acc)
            history['valid_loss'].append(valid_loss)
            print(f'train_acc: {train_acc}%, train_loss: {train_loss}, valid_acc: {valid_acc}%, valid_loss: {valid_loss}.')
            if early_stoping:
                es(valid_loss, self.sleep_net)
            scheduler.step()
        self.save_history(history, 'finetune')


    def test(self, test_loader, net_name=None):
        if net_name != None:
            path = os.path.join(self.save_path, net_name)
            net = torch.load(path)
        else:
            net = self.sleep_net
        net.to(self.device)
        net.eval()
        outputs = []
        targets = []
        for bt, (raw_inputs_bt, wave_inputs_bt, targets_bt) in enumerate(test_loader):
            with torch.no_grad():
                raw_inputs_bt = raw_inputs_bt.to(self.device)
                wave_inputs_bt = wave_inputs_bt.to(self.device)
                targets_bt = targets_bt.to(self.device)
                if len(targets_bt.shape) == 2:
                    targets_bt = targets_bt.contiguous().view(targets_bt.shape[0]*targets_bt.shape[1])
                outputs_bt = net(raw_inputs_bt, wave_inputs_bt)
                outputs.append(outputs_bt)
                targets.append(targets_bt)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        metrics = self.evaluate(targets, outputs)
        metrics.print_metrics()
        metrics.save_metrics(self.save_path, net_name)
        return metrics


    def test_res(self, test_loader):
        self.sleep_net.to(self.device)
        self.sleep_net.eval()
        outputs = []
        targets = []
        for bt, (raw_inputs_bt, wave_inputs_bt, targets_bt) in enumerate(test_loader):
            with torch.no_grad():
                raw_inputs_bt = raw_inputs_bt.to(self.device)
                wave_inputs_bt = wave_inputs_bt.to(self.device)
                targets_bt = targets_bt.to(self.device)
                if len(targets_bt.shape) == 2:
                    targets_bt = targets_bt.contiguous().view(targets_bt.shape[0]*targets_bt.shape[1])
                outputs_bt = self.sleep_net(raw_inputs_bt, wave_inputs_bt)
                outputs.append(outputs_bt)
                targets.append(targets_bt)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        return outputs, targets
