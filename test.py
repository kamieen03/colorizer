#!/usr/bin/env python3

from torchsummary import summary
import torch
import sys
from PIL import Image
from colorizer import Colorizer
import numpy as np

c = Colorizer()
c.eval()
c.load_state_dict(torch.load('colorizer.pth'))
img = np.asarray(Image.open('cat.jpg'))
A = c.rgb2Lab(img)[:1,:,:].unsqueeze(0).cuda()
with torch.no_grad():
    B = c(A)
    d = c.netD(torch.cat([A, B], 1))
print(c.criterionGAN(d, False))
print(c.criterionGAN(d, True))
print(d)
B = B.cpu()
grey = B[0][0].numpy()*255
Image.fromarray(grey).show()
rgb_B = c.Lab2rgb(B)
Image.fromarray(rgb_B).show()

