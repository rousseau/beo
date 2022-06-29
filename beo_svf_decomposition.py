#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:57:22 2022

@author: rousseau
"""
import os
from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl

import torchio as tio
import monai

import nibabel
import numpy as np

from beo_core_morph import SpatialTransformer, VecInt

field_file = home+'/Sync-Exp/Experiments/ants_1_on_60Warp.nii.gz'

field_image = nibabel.load(field_file)
field_data  = torch.Tensor(field_image.get_fdata())
field_data = torch.moveaxis(field_data, 4, 0)
field_data = torch.moveaxis(field_data, 4, 0)

batch_size = 1
trainloader = torch.utils.data.DataLoader(field_data, batch_size=batch_size)

class svf_model(pl.LightningModule):
  def __init__(self, shape, int_steps = 7, n_features = 8, sigma = 0.5):  
    super().__init__()  
    self.shape = shape
    self.transformer = SpatialTransformer(size=shape)
    self.int_steps = int_steps
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)
    self.n_features = n_features

    def double_conv(in_channels, out_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        #nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        #nn.Tanh(),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        #nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        #nn.Tanh(),
      )

    self.dc1 = double_conv(3, self.n_features)
    self.dc2 = double_conv(self.n_features, self.n_features)
    self.dc3 = double_conv(self.n_features, self.n_features)
    self.out = nn.Conv3d(self.n_features, 3, kernel_size=1)
  
  def forward(self, x):   
    x = self.dc1(x)
    x = self.dc2(x)
    x = self.dc3(x)
    return self.out(x)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    field = batch
    velocity = self(field)
    predicted_field = self.vecint(velocity)
    loss = F.mse_loss(field, predicted_field)
    return loss


n_epochs = 1000
in_shape = (field_data.shape[2],field_data.shape[3],field_data.shape[4])
n_features = 16
sigma = 0.01
int_steps = 5

net = svf_model(shape=in_shape, int_steps = int_steps, n_features = n_features, sigma = sigma)

trainer = pl.Trainer(gpus=1, max_epochs=n_epochs, logger=False)
trainer.fit(net, trainloader) 

svf = net.forward(field_data)

suffix = '_int'+str(int_steps)
suffix+= '_sigma'+str(sigma)
suffix+= '_nf'+str(n_features)

output_path = home+'/Sync-Exp/Experiments/'
svf_file = output_path+'svf'+suffix+'.nii.gz'
flow_file = output_path+'flow'+suffix+'.nii.gz'

o = tio.ScalarImage(tensor=svf[0].detach().numpy(), affine=field_image.affine)
o.save(svf_file)

flow = net.vecint(svf)
o = tio.ScalarImage(tensor=flow[0].detach().numpy(), affine=field_image.affine)
o.save(flow_file)