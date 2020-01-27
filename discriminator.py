import torch
from torch import nn

class PatchDiscriminator(nn.Module):
    def __init__(self, n=5, batch_norm=True, dropout=False):
        super(PatchDiscriminator, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        layers = []
        inc = 32 # one channel for L image and 3 for (generted or groundtruth) Lab image
        outc = 32
        layers.append(nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, True))
        for _ in range(n-1):
            layers.append(nn.Conv2d(inc, outc, kernel_size=4, stride=2, padding=1))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(outc))
            layers.append(nn.LeakyReLU(0.2, True))
            layers.append(nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(outc))
            layers.append(nn.LeakyReLU(0.2, True))
            inc = outc
            outc *= 2
        layers.append(nn.Conv2d(inc, inc, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Conv2d(inc, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)






