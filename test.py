#!/usr/bin/env python3

from torchsummary import summary
from generator import UNet
from discriminator import PatchDiscriminator
import sys



net = UNet().cuda()
summary(net, (1, int(sys.argv[1]), int(sys.argv[2])))
net2 = PatchDiscriminator().cuda()
summary(net2, (3, int(sys.argv[1]), int(sys.argv[2])))
