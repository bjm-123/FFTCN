# %%
import torch.optim as optim
from models.wavelet.network import *
from models.wavelet.parameter import Parameter
from models.base.model import BaseModel
from models.base.utils import EarlyStopping


# %%
class WaveletModel(BaseModel):
    def __init__(self, device, save_path, fold=None, name=None):
        BaseModel.__init__(self, device, save_path, fold, name)

        self.feature_net = WaveFeatureNet(Parameter.feature_net, Parameter.n_class)
        self.sleep_net = WaveSleepNet(Parameter.sleep_net, self.feature_net)

    # def _train(self, net, train_loader, epoch, n_epoch, optimizer:optim.Optimizer, alpha=0):
    #     net.train()
    #     acc = 0
    #     loss = 0
    #     sys.stdout.flush()
    #     sys.stdout.flush()
    #     counter = train_loader.dataset.all_counter
    #     weight = torch.tensor([counter[k] for k in sorted(counter.keys())])
    #     weight = torch.pow(torch.sum(weight)/weight, alpha)
    #     weight = weight.to(self.device)
    #     pbar = tqdm(train_loader, desc=f'EPOCH[{epoch+1}/{n_epoch}]', ncols=100)
    #     for bt, (inputs, targets) in enumerate(pbar):
    #         inputs = inputs.to(self.device)
    #         targets = targets.to(self.device)
    #         if len(targets.shape) == 2:
    #             targets = targets.contiguous().view(targets.shape[0]*targets.shape[1])

    #         outputs = net(inputs)
    #         losses = F.cross_entropy(outputs, targets, weight)
    #         # update loss metric
    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()

    #         acc_bt, loss_bt = self.evaluate(targets, outputs, losses)
    #         acc += acc_bt
    #         loss += loss_bt
    #         pbar.set_postfix({'acc': acc_bt, 'loss': loss_bt})
    #     return np.round(acc/len(train_loader)*100, 2), np.round(loss/len(train_loader), 4)

    # def _valid(self, net, valid_loader):
    #     net.eval()
    #     acc = 0
    #     loss = 0
    #     for bt, (inputs, targets) in enumerate(valid_loader):
    #         with torch.no_grad():
    #             inputs = inputs.to(self.device)
    #             targets = targets.to(self.device)
    #             if len(targets.shape) == 2:
    #                 targets = targets.contiguous().view(targets.shape[0]*targets.shape[1])
    #             outputs = net(inputs)
    #             losses = F.cross_entropy(outputs, targets)
    #         acc_bt, loss_bt = self.evaluate(targets, outputs, losses)
    #         acc += acc_bt
    #         loss += loss_bt
    #     return np.round(acc/len(valid_loader)*100, 2), np.round(loss/len(valid_loader), 4)

    def pretrain(self, train_loader, valid_loader, n_epoch, learn_rate, gamma):
        optimizer = optim.Adam(self.feature_net.parameters(), learn_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        history = {
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'valid_loss': []
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
            print(
                f'train_acc: {train_acc}%, train_loss: {train_loss}, valid_acc: {valid_acc}%, valid_loss: {valid_loss}.')
            scheduler.step()
        self.save_history(history, 'pretrain')

    def finetune(self, train_loader, valid_loader, n_epoch, learn_rate, lamb, gamma, alpha, early_stoping=False):
        f_params = list(map(id, self.feature_net.parameters()))
        s_params = filter(lambda p: id(p) not in f_params, self.sleep_net.parameters())
        optimizer = optim.Adam(
            params=[{'params': s_params}, {'params': self.feature_net.parameters(), 'lr': learn_rate * lamb}],
            lr=learn_rate
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        history = {
            'train_acc': [],
            'train_loss': [],
            'valid_acc': [],
            'valid_loss': []
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
            print(
                f'train_acc: {train_acc}%, train_loss: {train_loss}, valid_acc: {valid_acc}%, valid_loss: {valid_loss}.')
            if early_stoping:
                es(valid_loss, self.sleep_net)
            scheduler.step()
        self.save_history(history, 'finetune')

    # def test(self, test_loader, name=None):
    #     if name != None:
    #         path = os.path.join(self.save_path, name)
    #         net = torch.load(path)
    #     else:
    #         net = self.sleep_net
    #     net.to(self.device)
    #     net.eval()
    #     outputs = []
    #     targets = []
    #     for bt, (inputs_bt, targets_bt) in enumerate(test_loader):
    #         with torch.no_grad():
    #             inputs_bt = inputs_bt.to(self.device)
    #             targets_bt = targets_bt.to(self.device)
    #             if len(targets_bt.shape) == 2:
    #                 targets_bt = targets_bt.contiguous().view(targets_bt.shape[0]*targets_bt.shape[1])
    #             outputs_bt = net(inputs_bt)
    #             outputs.append(outputs_bt)
    #             targets.append(targets_bt)
    #     outputs = torch.cat(outputs)
    #     targets = torch.cat(targets)
    #     metrics = self.evaluate(targets, outputs)
    #     metrics.print_metrics()
    #     metrics.save_metrics(self.save_path)
    #     return metrics

    # def test_res(self, test_loader):
    #     self.sleep_net.to(self.device)
    #     self.sleep_net.eval()
    #     outputs = []
    #     targets = []
    #     for bt, (inputs_bt, targets_bt) in enumerate(test_loader):
    #         with torch.no_grad():
    #             inputs_bt = inputs_bt.to(self.device)
    #             targets_bt = targets_bt.to(self.device)
    #             if len(targets_bt.shape) == 2:
    #                 targets_bt = targets_bt.contiguous().view(targets_bt.shape[0]*targets_bt.shape[1])
    #             outputs_bt = self.sleep_net(inputs_bt)
    #             outputs.append(outputs_bt)
    #             targets.append(targets_bt)
    #     outputs = torch.cat(outputs)
    #     targets = torch.cat(targets)
    #     return outputs, targets

    # def evaluate(self, targets, outputs, losses = None):
    #     metrics = Metrics(outputs, targets, 5)
    #     if losses != None:
    #         return round(metrics.accuracy_score().item(), 4), round(losses.item(), 4)
    #     else:
    #         return metrics

    # def save(self):
    #     torch.save(self.sleep_net, os.path.join(self.save_path, 'network.pth'))

    # def save_history(self, history, name):
    #     np.savez(os.path.join(self.save_path, name+'his.npz'), **history)
    #     acc_train = history['train_acc']
    #     acc_val = history['valid_acc']
    #     loss_train = history['train_loss']
    #     loss_val = history['valid_loss']
    #     plt.figure(figsize=(20, 10))
    #     plt.subplot(121)
    #     plt.title('accuracy')    
    #     plt.plot(range(len(acc_train)),acc_train, label = 'train')
    #     plt.plot(range(len(acc_val)),acc_val, label = 'validation')
    #     plt.legend(loc='best')
    #     plt.subplot(122)
    #     plt.title('loss')
    #     plt.plot(range(len(loss_train)),loss_train, label = 'train')
    #     plt.plot(range(len(loss_val)),loss_val, label = 'validation')
    #     plt.legend(loc='best')
    #     plt.savefig(os.path.join(self.save_path, name+'his.png'))
    #     plt.close()
