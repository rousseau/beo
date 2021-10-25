#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import glob
import multiprocessing

from beo_torchio_datasets import get_dhcp



subjects = get_dhcp(max_subjects=100)

# DATA AUGMENTATION
normalization = tio.ZNormalization()
spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
transforms = [flip, spatial, normalization]
training_transform = tio.Compose(transforms)
validation_transform = tio.Compose([normalization])  

# SPLIT DATA
seed = 42  # for reproducibility
training_split_ratio = 0.9
num_subjects = len(subjects)
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects, generator=torch.Generator().manual_seed(seed))

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')

#%%
num_workers = 8
patch_size = 64
max_queue_length = 1024
samples_per_volume = 4
batch_size = 4
print('num_workers : '+str(num_workers))

probabilities = {0: 0, 1: 1}
sampler = tio.data.UniformSampler(patch_size)

patches_training_set = tio.Queue(
  subjects_dataset=training_set,
  max_length=max_queue_length,
  samples_per_volume=samples_per_volume,
  sampler=sampler,
  num_workers=num_workers,
  shuffle_subjects=True,
  shuffle_patches=True,
)

patches_validation_set = tio.Queue(
  subjects_dataset=validation_set,
  max_length=max_queue_length,
  samples_per_volume=samples_per_volume,
  sampler=sampler,
  num_workers=num_workers,
  shuffle_subjects=False,
  shuffle_patches=False,
)

training_loader_patches = torch.utils.data.DataLoader(
  patches_training_set, batch_size=batch_size)

validation_loader_patches = torch.utils.data.DataLoader(
  patches_validation_set, batch_size=batch_size)

#%%
class Unet(pl.LightningModule):
  def __init__(self, in_channels = 1, out_channels = 10, n_filters = 32, activation = 'relu'):
    super(Unet, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_features = n_filters
    self.activation = activation

    def double_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    self.dc1 = double_conv(self.in_channels, self.n_features)
    self.dc2 = double_conv(self.in_channels, self.n_features)
    self.dc3 = double_conv(self.in_channels, self.n_features)

    self.mp = nn.MaxPool3d(2)
    self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    self.out = nn.Conv3d(self.n_features, self.out_channels, kernel_size=1)

  def forward(self, x):
    #level 0 : original size
    #level 1 : downsampled by 2
    #level 2 : downsampled by 4

    x_l0 = x
    x_l1 = self.mp(x_l0)
    x_l2 = self.mp(x_l1)

    x_l2_conv = self.dc1(x_l2)
    x_l2_up = self.up(x_l2_conv)

    x_l1_conv = self.dc2(x_l1) + x_l2_up
    x_l1_up = self.up(x_l1_conv)

    x_l0_conv = self.dc3(x_l0) + x_l1_up

    xout = self.out(x_l0_conv)

    if self.activation == 'tanh':
      return nn.Tanh()(xout)
    elif self.activation == 'relu':
      return nn.ReLU()(xout)      
    elif self.activation == 'softmax':
      return nn.Softmax(dim=1)(xout)
    elif self.activation == 'gumbel':
      return nn.functional.gumbel_softmax(xout, hard=True)
    else:
      return xout   

  def evaluate_batch(self, batch):
    patches_batch = batch
    t1 = patches_batch['t1'][tio.DATA]
    t2 = patches_batch['t2'][tio.DATA]
    s = patches_batch['label'][tio.DATA]

    rt2 = self(t2)
    #bce = nn.BCEWithLogitsLoss()
    loss_recon = F.mse_loss(rt2,t2)
    #loss_seg = bce(st2,s)
    #loss_recon = F.mse_loss(rt2,t1)
    #loss_seg = 0

    return loss_recon
      
  def training_step(self, batch, batch_idx):
    loss = self.evaluate_batch(batch)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    loss = self.evaluate_batch(batch)
    self.log('val_loss', loss)
    self.log("hp_metric", loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer  

net = Unet(in_channels = 1, out_channels = 1, n_filters = 32, activation = 'relu')

prefix = 'unet_recon'
num_epochs = 5
output_path = home+'/Sync-Exp/Experiments/'

checkpoint_callback = ModelCheckpoint(
  dirpath=output_path,
  filename=prefix+'_{epoch:02d}', 
  verbose=True
  )
trainer = pl.Trainer(gpus=1, 
                      logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, name=prefix, log_graph=True),
                      max_epochs=num_epochs, 
                      progress_bar_refresh_rate=20, 
                      callbacks=[checkpoint_callback],
                      benchmark=True,
                      amp_level='01',
                      auto_scale_batch_size="binsearch")

trainer.fit(net, training_loader_patches, validation_loader_patches)
trainer.save_checkpoint(output_path+prefix+'.ckpt')  
print('Finished Training')                      