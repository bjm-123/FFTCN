# %%
import torch.optim as optim

from models.raw.network import *
from models.raw.parameter import Parameter
from models.base.utils import EarlyStopping
from models.base.model import BaseModel
# %%
class RawModel(BaseModel):
    def __init__(self, device, save_path, fold=None, name=None):
        BaseModel.__init__(self, device, save_path, fold, name)

        self.feature_net = RawFeatureNet(Parameter.feature_net, Parameter.n_class)
        self.sleep_net = RawSleepNet(Parameter.sleep_net, self.feature_net)


    def pretrain(self, train_loader, valid_loader, n_epoch, learn_rate, gamma):
        optimizer = optim.Adam(self.feature_net.parameters(), learn_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        history = {
            'train_acc':[],
            'train_loss':[],
            'valid_acc':[],
            'valid_loss':[]
        }
        self.feature_net.to(self.device)
        self.feature_net.pretrain()
        for epoch in range(n_epoch):
            train_acc, train_loss = self._train(self.feature_net, train_loader, epoch, n_epoch, optimizer)
            valid_acc, valid_loss = self._valid(self.feature_net, valid_loader)
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['valid_acc'].append(valid_acc)
            history['valid_loss'].append(valid_loss)
            print(f'train_acc: {train_acc}%, train_loss: {train_loss}, valid_acc: {valid_acc}%, valid_loss: {valid_loss}.')
            scheduler.step()
        self.save_history(history, 'pretrain')


    def finetune(self, train_loader, valid_loader, n_epoch, learn_rate, lamb, gamma, alpha, early_stoping=False):
        f_params = list(map(id, self.feature_net.parameters()))
        s_params = filter(lambda p: id(p) not in f_params, self.sleep_net.parameters())
        optimizer = optim.Adam(
            params=[{'params': s_params}, {'params': self.feature_net.parameters(), 'lr': learn_rate*lamb}],
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
        self.feature_net.finetune()
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

