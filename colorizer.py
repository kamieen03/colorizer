import torch
import cv2

from generator import UNet
from discriminator import PatchDiscriminator

class Colorizer(nn.Model):
    def __init__(self, TRAIN=True):
        super(Colorizer, self).__init__()
        # define networks (both generator and discriminator)
        self.netG = UNet()

        if TRAIN:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = PatchDiscriminator()

            # define loss functions
            self.criterionGAN = GANLoss.cuda()
            self.criterionL1 = torch.nn.L1Loss().cuda()
            # and optimizers
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), 1e-4, momentum=0.9)
            self.optimizer_D = torch.optim.SGD(self.netD.parameters(), 1e-4, momentum=0.9)


    def forward(self, A):
        return self.netG(A)

    def backward_D(self, A, B, fake_b):
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((A, fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((A, B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) / 2.0
        self.loss_D.backward()

    def backward_G(self, A, B, fake_b):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((A, fake_b), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(B, fake_b) * self.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self, batch):
        B = self.rgb2Lab(batch).cuda()
        A = B[:,:1,:,:].clone()
        fake_b = self.forward(A)                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D(A, B, fake_b)                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G(A, B, fake_b)                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


    def validate_G(self, batch):
        """Calculate GAN and L1 loss for the generator"""
        B = self.rgb2Lab(batch)
        A = B[:,:1,:,:]
        fake_b = self.forward(A)                   # compute fake images: G(A)

        fake_AB = torch.cat((A, fake_b), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(B, fake_b) * self.lambda_L1
        # combine loss and calculate gradients
        return self.loss_G_GAN + self.loss_G_L1

    def rgb2Lab(batch):
        '''
        Accepts np array in uint8 RGB BxHxWxC format.
        Returns torch tensor in float32 Lab BxCxHxW format.
        '''

        b, h, w, c = batch.shape
        assert c == 3
        Lab_batch = np.zeros((b, h, w, 3))

        for i in range(b):
            Lab_batch[i] = cv2.cvtColor(batch[i], cv2.COLOR_RGB2LAB)
        Lab_batch = Lab_batch.transpose(0,3,1,2)
        assert Lab_batch.shape[1] == 3
        Lab_batch = torch.from_numpy(Lab_batch) / 255.0 #implicit coversion uint8 -> float32

        return Lab_batch

    def Lab2rgb(img):
        '''
        Accepts torch tensor in float32 Lab CxHxW format.
        Returns np array in uint8 RGB HxWxC format. 
        '''
        img = img.transpose(1,2,0) * 255
        img = img.astype('uint8')
        img = cv2.cvtColor(img, cv2.LAB2RGB)
        return img

class GANLoss(torch.nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = torch.nnBCEWithLogitsLoss()

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = torch.ones(prediction.shape)
        else:
            target_tensor = torch.zeros(prediction.shape)
        return self.loss(prediction, target_tensor)

