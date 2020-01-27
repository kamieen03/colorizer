import torch
import cv2
import numpy as np
from torchsummary import summary

from generator import UNet
from discriminator import PatchDiscriminator

class Colorizer(torch.nn.Module):
    def __init__(self, TRAIN=True):
        super(Colorizer, self).__init__()
        self.netG = UNet().cuda()
        self.lambda_L1 = 100

        if TRAIN:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = PatchDiscriminator().cuda()

            # define loss functions
            self.criterionGAN = GANLoss().cuda()
            self.criterionL1 = torch.nn.L1Loss().cuda()
            # and optimizers
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), 1e-3, momentum=0.9)
            self.optimizer_D = torch.optim.SGD(self.netD.parameters(), 1e-3, momentum=0.9)
            #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1e-4)
            #self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=1e-4)

            summary(self.netG, (1,400,400))
            summary(self.netD, (4,400,400))


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
        B = self.rgb2Lab(batch).cuda()
        A = B[:,:1,:,:].clone()
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
        B = self.rgb2Lab(batch).cuda()
        A = B[:,:1,:,:]
        fake_b = self.forward(A)                   # compute fake images: G(A)

        fake_AB = torch.cat((A, fake_b), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(B, fake_b) * self.lambda_L1
        # combine loss and calculate gradients
        return self.loss_G_GAN + self.loss_G_L1

    def rgb2Lab(self, batch):
        '''
        Accepts torch tensor in float32 RGB BxCxHxW format.
        Returns torch tensor in float32 Lab BxCxHxW format.
        '''
        if len(batch.shape) == 4:
            batch = batch.numpy().transpose(0, 2, 3, 1)
            b, h, w, c = batch.shape
            assert c == 3
            Lab_batch = np.zeros((b, h, w, 3))

            for i in range(b):
                Lab_batch[i] = cv2.cvtColor(batch[i], cv2.COLOR_RGB2LAB)
            Lab_batch = Lab_batch.transpose(0,3,1,2)
            assert Lab_batch.shape[1] == 3
        else:
            Lab_batch = cv2.cvtColor(batch, cv2.COLOR_RGB2LAB)
            Lab_batch = Lab_batch.transpose(2,0,1)

        Lab_batch = torch.from_numpy(Lab_batch) / 255.0
        Lab_batch = Lab_batch.float()

        return Lab_batch

    def Lab2rgb(self, img):
        '''
        Accepts torch tensor in float32 Lab CxHxW format.
        Returns np array in uint8 RGB HxWxC format. 
        '''
        img = img[0].numpy()*255
        img = img.transpose(1,2,0)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        img = img * 255
        img = img.astype('uint8')
        return img

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

