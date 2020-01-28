#!/usr/bin/env python3

from torchsummary import summary
import torch
import numpy as np
from loader import Dataset
from colorizer import Colorizer

BATCH_SIZE = 32
CROP_SIZE = 400
EPOCHS = 50
MODEL_SAVE_PATH = 'colorizer.pth'

class Trainer:
    def __init__(self):
        self.train_set = self.load_dataset('data/train')
        self.val_set = self.load_dataset('data/val')
        self.model = Colorizer().cuda()
        #self.model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    def load_dataset(self, path):
        """Load the datasets"""
        dataset = Dataset(path, CROP_SIZE)
        loader = torch.utils.data.DataLoader(dataset     = dataset,
                                             batch_size  = BATCH_SIZE,
                                             shuffle     = True,
                                             num_workers = 8,
                                             drop_last   = True)
        return loader

    def train(self):
        best_val = 1e9
        with open('train_log.txt', 'w+') as f:
            for epoch in range(1, EPOCHS+1): # count from one
                self.train_single_epoch(epoch, f)
                val = self.validate_single_epoch(epoch, f)
        #        self.model.lr_scheduler_G.step(val)
        #        self.model.lr_scheduler_D.step(val)
                best_val = val
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)


    def train_single_epoch(self, epoch, f):
        batch_num = len(self.train_set)      # number of batches in training epoch
        self.model.netG.train()
        self.model.netD.train()

        for num, batch in enumerate(self.train_set):
            loss_G_GAN, loss_G_L1, loss_D = self.model.optimize_parameters(batch)
            log = f'Train Epoch: [{epoch}/{EPOCHS}] ' + \
                  f'Batch: [{num+1}/{batch_num}] ' + \
                  f'Loss G_GAN: {loss_G_GAN:.6f} ' + \
                  f'Loss G_L1: {loss_G_L1:.6f} ' + \
                  f'Loss D: {loss_D:.6f}' 
            print(log)
            f.write(log+'\n')

    def validate_single_epoch(self, epoch, f):
        batch_num = len(self.val_set)      # number of batches in training epoch
        self.model.netG.eval()
        self.model.netD.eval()
        losses = []

        with torch.no_grad():
            for num, batch in enumerate(self.val_set):
                loss = self.model.validate_G(batch)
                losses.append(loss.item())
                log = f'Validate Epoch: [{epoch}/{EPOCHS}] ' + \
                      f'Batch: [{num+1}/{batch_num}] ' + \
                      f'Loss: {loss:.6f}'
                print(log)
                f.write(log+'\n')
        return float(np.mean(np.array(losses)))


def main():
    t = Trainer()
    t.train()

if __name__ == '__main__':
    main()
