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
import pytorch_lightning as pl

import torchio as tio
import monai

import nibabel
import numpy as np

from beo_core_morph import SpatialTransformer, VecInt, Grad3d, NCC, MSE, Unet, GaussianSmoothing

#%%



class Toto(nn.Module):
  def __init__(self, n_channels = 1, n_classes = 3, n_features = 8, sigma=0.5):
    super(Toto, self).__init__()

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.n_features = n_features
    self.smoothing = GaussianSmoothing(channels=n_classes, kernel_size=5,sigma=sigma, dim=3)

    def double_conv(in_channels, out_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
      )

    self.dc1 = double_conv(self.n_channels, self.n_features)
    self.dc2 = double_conv(self.n_features*2, self.n_features)
    
    self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

  def forward(self, x, y):

    #extract features independently
    fx = self.dc1(x)
    fy = self.dc1(y)

    #concatenate
    f = torch.cat([fx,fy],dim=1)

    #estimate field
    field = self.out(self.dc2(f))

    return self.smoothing(field)






#%%

class meta_registration_model(pl.LightningModule):
  def __init__(self, shape, int_steps = 7, n_features = 8, sigma = 0.5):  
    super().__init__()  
    self.shape = shape
    self.toto = Toto(n_features = n_features, sigma = sigma)
    self.transformer = SpatialTransformer(size=shape)
    self.int_steps = int_steps
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)

    self.lambda_similarity = 1
    self.lambda_grad_flow  = 0.1
    self.lambda_magn_flow  = 0.01
    self.lambda_grad_velocity  = 0.1
    self.lambda_magn_velocity  = 0.01    
    self.bidir = True
    self.similarity = monai.losses.LocalNormalizedCrossCorrelationLoss() #NCC() #MSE()

    self.ap = nn.AvgPool3d(2)
    self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

  def forward(self,source,target):
    #scale 0 : original size
    #scale 1 : downsampling by 2, etc.

    y_source = []
    y_target = []
    
    s_s1 = self.ap(source)
    t_s1 = self.ap(target)

    s_s2 = self.ap(s_s1)
    t_s2 = self.ap(t_s1)

    #flow scale 2 at original size
    fv_s2 = self.up(self.up(self.toto(s_s2,t_s2)))
    bv_s2 = -fv_s2
    ff_s2 = self.vecint(fv_s2)
    bf_s2 = self.vecint(bv_s2)    

    y_source.append(self.transformer(source, ff_s2))
    y_target.append(self.transformer(target, bf_s2))

    warped_source_s1 = self.ap(y_source[0])
    fv_s1 = self.up(self.toto( warped_source_s1,t_s1))
    warped_target_s1 = self.ap(y_target[0])
    bv_s1 = self.up(self.toto( warped_target_s1,s_s1))    
    fv_s1 = (fv_s1 - bv_s1)/2.0

    fv_s1 = fv_s2 + self.transformer(fv_s1,fv_s2)
    bv_s1 = -fv_s1
    ff_s1 = self.vecint(fv_s1)
    bf_s1 = self.vecint(bv_s1)    

    y_source.append(self.transformer(source, ff_s1))
    y_target.append(self.transformer(target, bf_s1))    

    fv_s0 = self.toto( y_source[1], target )
    bv_s0 = self.toto( y_target[1], source )

    fv_s0 = (fv_s0 - bv_s0)/2.0

    forward_velocity = fv_s1 + self.transformer(fv_s0,fv_s1)
    backward_velocity = -forward_velocity
    forward_flow = self.vecint(forward_velocity)
    backward_flow= self.vecint(backward_velocity) 

    y_source.append(self.transformer(source, forward_flow))
    y_target.append(self.transformer(target, backward_flow))

    return y_source, y_target, forward_velocity, forward_flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source, target = batch
    y_source,y_target, forward_velocity, forward_flow = self(source,target)

    loss = 0
    for i in range(len(y_source)):      
      if self.bidir is True:
        loss = self.lambda_similarity * (self.similarity.forward(target,y_source[i]) + self.similarity.forward(y_target[i],source))/2
      else:
        loss = self.lambda_similarity * self.similarity.forward(target,y_source[i])
         
    if self.lambda_grad_flow > 0:
      loss += self.lambda_grad_flow * Grad3d().forward(forward_flow) 

    if self.lambda_magn_flow > 0:  
      loss += self.lambda_magn_flow * F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)    
    
    if self.lambda_grad_velocity > 0:
      loss += self.lambda_grad_velocity * Grad3d().forward(forward_velocity) 

    if self.lambda_magn_velocity > 0:  
      loss += self.lambda_magn_velocity * F.mse_loss(torch.zeros(forward_velocity.shape,device=self.device),forward_velocity)  
    
    return loss 

#%%

from torch.utils.data import Dataset
class CustomDataSet(Dataset):
  def __init__(self, X):
    self.X = X
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    index_source = torch.randint(self.len,(1,))
    index_target = torch.randint(self.len,(1,))

    _source = self.X[index_source][0]
    _target = self.X[index_target][0]
    
    return _source, _target  
  
from torch.utils.data import Dataset
class TwoDataSet(Dataset):
  def __init__(self, X):
    self.X = X
    self.len = len(self.X)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    index_source = 0
    index_target = 1

    _source = self.X[index_source]
    _target = self.X[index_target]
    
    return _source, _target    

#%%
data_path = home+'/Sync-Exp/Experiments/Atlas_dhcp/'

#target_file = data_path+'sub-CC00053XX04_ses-8607_T2w.flirt.nii.gz' 
#source_file = data_path+'sub-CC00052XX03_ses-8300_T2w.flirt.nii.gz'
 
#target_file = home+'/Sync-Exp/Experiments/6Months-T1w_rs.nii.gz'
#source_file = home+'/Sync-Exp/Experiments/1Month-T1w_rs.nii.gz'

target_file = home+'/Sync-Exp/Experiments/atlas_unet3template0_rs.nii.gz'
source_file = home+'/Sync-Exp/Experiments/sub_T07_static_3DT1_flirt_rs.nii.gz'

target_image = nibabel.load(target_file)
source_image = nibabel.load(source_file)

target_data  = torch.Tensor(target_image.get_fdata())
source_data  = torch.Tensor(source_image.get_fdata())

target_data = torch.unsqueeze(torch.unsqueeze(target_data, 0),0)
source_data = torch.unsqueeze(torch.unsqueeze(source_data, 0),0)

x_train = torch.cat([source_data,target_data],dim=0)
#%%
batch_size_reg = 1

trainset_reg = TwoDataSet(x_train)
trainloader_reg = torch.utils.data.DataLoader(trainset_reg, batch_size=batch_size_reg)   

#%%

n_epochs_reg = 100
in_shape = (x_train.shape[2],x_train.shape[3],x_train.shape[4])
n_features = 16
sigma = 1

#reg_net = meta_registration_model(shape=in_shape, n_features = n_features)
reg_net = meta_registration_model(shape=in_shape, int_steps = 7, n_features = n_features, sigma = sigma)

trainer_reg = pl.Trainer(gpus=1, max_epochs=n_epochs_reg, logger=False, precision=16)
trainer_reg.fit(reg_net, trainloader_reg)  
#%%

y_source, y_target, forward_velocity, forward_flow = reg_net.forward(source_data,target_data)
#y_source = reg_net.forward(source_data,target_data)

suffix = '_lncc_toto'
suffix+= '_mag'+str(reg_net.lambda_magn_flow)
suffix+= '_grad'+str(reg_net.lambda_grad_flow)
suffix+= '_epoch'+str(n_epochs_reg)
suffix+= '_sigma'+str(sigma)
suffix+= '_int'+str(reg_net.int_steps)
suffix+= '_gradv'+str(reg_net.lambda_grad_velocity)
suffix+= '_magv'+str(reg_net.lambda_magn_velocity)
if reg_net.bidir is True:
  suffix+= '_bidir'

y_source = y_source[2]
y_target = y_target[2]

output_path = home+'/Sync-Exp/Experiments/'
y_target_file = output_path+'y_target'+suffix+'.nii.gz'
y_source_file = output_path+'y_source'+suffix+'.nii.gz'
velocity_file = output_path+'velocity'+suffix+'.nii.gz'
flow_file = output_path+'flow'+suffix+'.nii.gz'

o = tio.ScalarImage(tensor=y_source[0].detach().numpy(), affine=source_image.affine)
o.save(y_source_file)
o = tio.ScalarImage(tensor=y_target[0].detach().numpy(), affine=target_image.affine)
o.save(y_target_file)
o = tio.ScalarImage(tensor=forward_velocity[0].detach().numpy(), affine=source_image.affine)
o.save(velocity_file)
o = tio.ScalarImage(tensor=forward_flow[0].detach().numpy(), affine=source_image.affine)
o.save(flow_file)

#trainer_reg.save_checkpoint(output_path+'model.ckpt')

