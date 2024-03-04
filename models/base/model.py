# %%
import os
import sys

import matplotlib
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import datetime
import matplotlib.pyplot as plt
from models.base.parameter import Parameter
from models.base.utils import Metrics
matplotlib.use('TkAgg')


# %%
class BaseModel:
    def __init__(self, device, save_path, fold=None, name=None):

        self.device = device
        self.save_path = save_path
        if name is not None:
            self.save_path = os.path.join(self.save_path, self.__class__.__name__, str(name))
        else:
            now = datetime.datetime.now()
            now_str = now.strftime('%m.%d %H.%M')
            self.save_path = os.path.join(self.save_path, self.__class__.__name__, now_str)

        if fold is not None:
            self.save_path = os.path.join(self.save_path, str(fold))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _train(self, net, train_loader, epoch, n_epoch, optimizer: optim.Optimizer, alpha=0):
        net.train()
        acc = 0
        loss = 0
        sys.stdout.flush()
        counter = train_loader.dataset.all_counter
        weight = torch.tensor([counter[k] for k in sorted(counter.keys())])
        weight = torch.pow(torch.sum(weight) / weight, alpha)
        weight = weight.to(self.device)
        pbar = tqdm(train_loader, desc=f'EPOCH[{epoch + 1}/{n_epoch}]', ncols=100)
        for bt, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if len(targets.shape) == 2:
                targets = targets.contiguous().view(targets.shape[0] * targets.shape[1])

            outputs = net(inputs)
            losses = F.cross_entropy(outputs, targets, weight)
            # update loss metric
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            acc_bt, loss_bt = self.evaluate(targets, outputs, losses)
            acc += acc_bt
            loss += loss_bt
            pbar.set_postfix({'acc': acc_bt, 'loss': loss_bt})
        return np.round(acc / len(train_loader) * 100, 2), np.round(loss / len(train_loader), 4)

    def _valid(self, net, valid_loader):
        net.eval()
        acc = 0
        loss = 0
        for bt, (inputs, targets) in enumerate(valid_loader):
            with torch.no_grad():
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if len(targets.shape) == 2:
                    targets = targets.contiguous().view(targets.shape[0] * targets.shape[1])
                outputs = net(inputs)
                losses = F.cross_entropy(outputs, targets)
            acc_bt, loss_bt = self.evaluate(targets, outputs, losses)
            acc += acc_bt
            loss += loss_bt
        return np.round(acc / len(valid_loader) * 100, 2), np.round(loss / len(valid_loader), 4)

    def test(self, test_loader, net_name=None):
        if net_name is not None:
            path = os.path.join(self.save_path, net_name)
            net = torch.load(path)
        else:
            net = self.sleep_net
        net.to(self.device)
        net.eval()
        outputs = []
        targets = []
        for bt, (inputs_bt, targets_bt) in enumerate(test_loader):
            with torch.no_grad():
                inputs_bt = inputs_bt.to(self.device)
                targets_bt = targets_bt.to(self.device)
                if len(targets_bt.shape) == 2:
                    targets_bt = targets_bt.contiguous().view(targets_bt.shape[0] * targets_bt.shape[1])
                outputs_bt = net(inputs_bt)
                outputs.append(outputs_bt)
                targets.append(targets_bt)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        metrics = self.evaluate(targets, outputs)
        metrics.print_metrics()
        metrics.save_metrics(self.save_path, net_name)
        return metrics

    def test_res(self, test_loader, net_name=None):
        if net_name is not None:
            path = os.path.join(self.save_path, net_name)
            net = torch.load(path)
        else:
            net = self.sleep_net
        net.to(self.device)
        net.eval()
        outputs = []
        targets = []
        for bt, (inputs_bt, targets_bt) in enumerate(test_loader):
            with torch.no_grad():
                inputs_bt = inputs_bt.to(self.device)
                targets_bt = targets_bt.to(self.device)
                if len(targets_bt.shape) == 2:
                    targets_bt = targets_bt.contiguous().view(targets_bt.shape[0] * targets_bt.shape[1])
                outputs_bt = net(inputs_bt)
                outputs.append(outputs_bt)
                targets.append(targets_bt)
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        return outputs, targets

    def evaluate(self, targets, outputs, losses=None):
        metrics = Metrics(outputs, targets, Parameter.n_class)
        if losses is not None:
            return round(metrics.accuracy_score().item(), 4), round(losses.item(), 4)
        else:
            return metrics

    def save(self):
        torch.save(self.sleep_net, os.path.join(self.save_path, 'network.pth'))

    def save_history(self, history, name):
        np.savez(os.path.join(self.save_path, name + 'his.npz'), **history)
        acc_train = history['train_acc']
        acc_val = history['valid_acc']
        loss_train = history['train_loss']
        loss_val = history['valid_loss']
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.title('accuracy')
        plt.plot(range(len(acc_train)), acc_train, label='train')
        plt.plot(range(len(acc_val)), acc_val, label='validation')
        plt.legend(loc='best')
        plt.subplot(122)
        plt.title('loss')
        plt.plot(range(len(loss_train)), loss_train, label='train')
        plt.plot(range(len(loss_val)), loss_val, label='validation')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.save_path, name + 'his.png'))
        plt.close()
