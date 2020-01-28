import torch
from torch.nn import init
import cv2
import numpy as np
from torchsummary import summary

from generator import UNet
from discriminator import PatchDiscriminator

class Colorizer(torch.nn.Module):
    def __init__(self, TRAIN=True):
        super(Colorizer, self).__init__()
        self.netG = UNet()
        self.lambda_L1 = 100

        if TRAIN:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netG.cuda()
            self.netD = PatchDiscriminator().cuda()
            self._init_weights(self.netG)
            self._init_weights(self.netD)

            # define loss functions
            self.criterionGAN = GANLoss().cuda()
            self.criterionL1 = torch.nn.L1Loss().cuda()
            # and optimizers
            #self.optimizer_G = torch.optim.SGD(self.netG.parameters(), 1e-3, momentum=0.9)
            #self.optimizer_D = torch.optim.SGD(self.netD.parameters(), 1e-3, momentum=0.9)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), betas=(0.5, 0.999), lr=2e-4)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), betas=(0.5, 0.999), lr=2e-4)
            self.lr_scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_G, factor=0.3, patience=3) 
            self.lr_scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_D, factor=0.3, patience=3) 

            summary(self.netG, (1,400,400))
            summary(self.netD, (3,400,400))


    def forward(self, A):
        return self.netG(A)

    def backward_D(self, A, B, fake_b):
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((A, fake_b), 1)
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((A, B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) / 2.0
        loss_D.backward()
        return loss_D

    def backward_G(self, A, B, fake_b):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((A, fake_b), 1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = self.criterionL1(B, fake_b) * self.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        return loss_G_GAN, loss_G_L1

    def optimize_parameters(self, batch):
        A, B = batch
        A = A.cuda()
        B = B.cuda()
        #print(A.mean(), A.std())
        #print(B[:,0,:,:].mean(), B[:,0,:,:].std())
        #print(B[:,1,:,:].mean(), B[:,1,:,:].std())
        fake_b = self.forward(A)                   # compute fake images: G(A)
        # update D
        for param in self.netD.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        loss_D = self.backward_D(A, B, fake_b)                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        for param in self.netD.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        loss_G_GAN, loss_G_L1 = self.backward_G(A, B, fake_b)                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        return loss_G_GAN, loss_G_L1, loss_D


    def validate_G(self, batch):
        """Calculate GAN and L1 loss for the generator"""
        A, B = batch
        A.cuda(); B.cuda()
        fake_b = self.forward(A)                   # compute fake images: G(A)

        fake_AB = torch.cat((A, fake_b), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(B, fake_b) * self.lambda_L1
        # combine loss and calculate gradients
        return self.loss_G_GAN + self.loss_G_L1

    def Lab2rgb(self, L, AB):
        """Convert an Lab tensor image to a RGB numpy output
        Parameters:
            L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
            AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)
        Returns:
            rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
        """
        AB2 = AB * 110.0
        L2 = (L + 1.0) * 50.0
        Lab = torch.cat([L2, AB2], dim=1)
        Lab = Lab[0].data.cpu().float().numpy()
        Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
        rgb = color.lab2rgb(Lab) * 255
        return rgb

    def _init_weights(self, net, init_type='kaiming', init_gain=0.02):
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        net.apply(init_func)

class GANLoss(torch.nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.target_set = False

    def __call__(self, prediction, target_is_real):
        if not self.target_set:
            self.target_set = True
            self.target_ones = torch.ones(prediction.shape).cuda()
            self.target_zeros = torch.zeros(prediction.shape).cuda()
        if target_is_real:
            target_tensor = self.target_ones
        else:
            target_tensor = self.target_zeros
        return self.loss(prediction, target_tensor)

