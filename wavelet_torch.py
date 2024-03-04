# %%
from tqdm import tqdm
import os
import torch
import math
import numpy as np


# %%
class cwt:
    def __init__(self, dt, n, dj=0.25, J=30, param=6, mother='morlet', device=torch.device("cuda:0")) -> None:
        self.n = n
        self.device = device

        self.scale = 2 * dt * 2. ** (torch.arange(0, J, device=device) * dj)
        self.scale = torch.unsqueeze(self.scale, 1)

        fourier_factor = 4 * math.pi / (param + math.sqrt(2 + param ** 2))
        self.freq = 1. / (fourier_factor * self.scale)
        self.period = 1. / self.freq

        kplus = torch.arange(1, int(n / 2) + 1)
        kplus = kplus * 2 * math.pi / (n * dt)
        kminus = torch.arange(1, int((n - 1) / 2) + 1)
        kminus, _ = torch.sort((-kminus * 2 * math.pi / (n * dt)))
        k = torch.cat((torch.tensor([0.]), kplus, kminus))
        self.k = torch.unsqueeze(k, 0)
        self.k = self.k.to(device)
        self.kp = (self.k > 0.).float()

        expnt = -(self.scale * self.k - param) ** 2 / 2. * self.kp
        norm = torch.sqrt(self.scale * self.k[0, 1]) * (math.pi ** (-0.25)) * math.sqrt(n)
        daughter = norm * torch.exp(expnt)
        self.daughter = daughter * self.kp  # Heaviside step function

    def __call__(self, x, length):
        x = torch.tensor(x, device=self.device)
        f = torch.fft.fft(x)
        wave = torch.fft.ifft(f * self.daughter)
        spectrum = torch.log2(torch.abs(wave) ** 2 + 1e-10)
        return self.compress(spectrum, length)

    def compress(self, data, length=60):
        shape = list(data.shape)
        shape[-1] = length
        shape.append(-1)
        data = torch.reshape(data, shape)
        exp = torch.mean(data, dim=-1)
        max, _ = torch.max(exp, dim=-1, keepdim=True)
        max, _ = torch.max(max, dim=-2, keepdim=True)
        min, _ = torch.min(exp, dim=-1, keepdim=True)
        min, _ = torch.min(min, dim=-2, keepdim=True)
        exp = (exp - min) / (max - min)
        # var = torch.var(data, axis = -1)
        # var = (var-torch.min(var))/(torch.max(var)-torch.min(var))
        exp = torch.unsqueeze(exp, -3)
        return exp


# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    device = torch.device("cpu")
    source_path = r'D:\BJM\SHHS-1'
    target_path = r'D:\BJM\SHHS-clean'

    dirs = [os.listdir(source_path)[0]]
    pbar = tqdm(dirs, ncols=100)
    for i, dir in enumerate(pbar):
        with np.load(os.path.join(source_path, dir)) as f:
            data = f['x']
            label = f['y']
        # sample_n = len(data) // 20
        # data = data[:sample_n*20].reshape((-1, 20, *data.shape[1:]))
        for d in data:
            wavelet = cwt(1 / 100, 3000, device=device)
            spectrum = wavelet(d, 120)
            a = spectrum.numpy()
        if (spectrum.isnan()).any():
            print(dir + ' nan')
            plt.contourf(spectrum[0, 0])
            plt.show()

# %%
