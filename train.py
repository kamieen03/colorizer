#!/usr/bin/env python3

from torchsummary import summary
from loader import Dataset
from colorizer import Colorizer

BATCH_SIZE = 16
CROP_SIZE = 400
EPOCHS = 2
MODEL_SAVE_PATH = 'colorizer.pth'

class Trainer:
    def __init__():
        self.train_set = self.laod_dataset('data/train')
        self.val_set = self.laod_dataset('data/val')
        self.model = Colorizer().cuda()

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
                if val < best_val:
                    best_val = val
                    torch.save(self.model.state_dict(), MODEL_SAVE_PATH)


    def train_single_epoch(self, epoch, f):
        batch_num = len(self.train_set)      # number of batches in training epoch
        self.model.train()

        for num, batch in enumerate(self.train_set):
            self.model.optimie_parameters(batch)
            log = f'Train Epoch: [{epoch}/{EPOCHS}] ' +
                  f'Batch: [{num+1}/{batch_num}] ' +
                  f'Loss: {loss:.6f}'
            print(log)
            f.write(log)

    def validate_single_epoch(self, epoch, f):
        batch_num = len(self.valid_set)      # number of batches in training epoch
        self.model.eval()
        losses = []

        with torch.no_grad():
            for num, batch in enumerate(self.valid_set):
                loss = self.model.validate_G(batch)
                losses.append(loss.item())
                log = f'Validate Epoch: [{epoch}/{EPOCHS}] ' +
                      f'Batch: [{num+1}/{batch_num}] ' +
                      f'Loss: {loss:.6f}'
                print(log)
                f.write(log)
        return np.mean(np.array(losses))

