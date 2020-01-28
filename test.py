#!/usr/bin/env python3

from torchsummary import summary
import torch
import sys
from PIL import Image
from colorizer import Colorizer
import numpy as np

num = 1
if len(sys.argv) > 1:
    num = int(sys.argv[1])

c = Colorizer()
c.load_state_dict(torch.load('colorizer.pth'))
c.eval()

img = np.asarray(Image.open(f'cat{num}.jpg'))
L, _ = c.rgb2Lab(img)
L = L.unsqueeze(0).cuda()
with torch.no_grad():
    AB = c(L)
    d = c.netD(torch.cat([L, AB], 1))
print(d)
print('Loss', c.criterionGAN(d, False))

rgb_B = c.Lab2rgb(L, AB)
Image.fromarray(rgb_B).show()

