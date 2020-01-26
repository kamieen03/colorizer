import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, nfc=32):
        super(UNet, self).__init__()
        self.batch_norm = True
        self.dropout = False

        self.down1 = Down(1, 1*nfc, self.batch_norm, self.dropout)
        self.down2 = Down(1*nfc, 2*nfc, self.batch_norm, self.dropout)
        self.down3 = Down(2*nfc, 4*nfc, self.batch_norm, self.dropout)
        self.down4 = Down(4*nfc, 8*nfc, self.batch_norm, self.dropout)
        self.down5 = Down(8*nfc, 8*nfc, self.batch_norm, self.dropout)
        self.up1   = Up(16*nfc, 4*nfc, self.batch_norm, self.dropout)
        self.up2   = Up(8*nfc, 2*nfc, self.batch_norm, self.dropout)
        self.up3   = Up(4*nfc, 1*nfc, self.batch_norm, self.dropout)
        self.up4   = Up(2*nfc, 1*nfc, self.batch_norm, self.dropout)
        self.outc  = SingleUp(1*nfc, 3, self.batch_norm, self.dropout)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class Down(nn.Module):
    def __init__(self, inc, outc, batch_norm, dropout):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(outc)
        self.r1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(outc)
        self.r2 = nn.LeakyReLU(0.2, True)
        
    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.r1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.r2(x)
        return x


class Up(nn.Module):
    def __init__(self, inc, outc, batch_norm, dropout):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.up = nn.ConvTranspose2d(inc // 2, inc // 2, kernel_size=3, stride=2)

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(outc)
        self.r1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(outc)
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        b1,c1,h1,w1 = x1.size()
        b2,c2,h2,w2 = x2.size()
        diffY = torch.tensor([h2 - h1])
        diffX = torch.tensor([w2 - w1])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.r1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.r2(x)
        return x


class SingleUp(nn.Module):
    def __init__(self, inc, outc, batch_norm, dropout):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(outc)
        self.r1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(outc)
        self.r2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.r1(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.r2(x)
        return x
