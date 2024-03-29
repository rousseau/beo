#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import argparse

from beo_core_morph import SpatialTransformer, VecInt, Grad3d, NCC, MSE, Unet, GaussianSmoothing, conv_bn_relu

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

class velocity_model(nn.Module):
  def __init__(self, in_channels = 1, scale_factor=1, n_features=32):
    super(velocity_model, self).__init__()

    self.n_features = n_features
    self.ap = nn.AvgPool3d(scale_factor)
    self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)

    self.cbr_1 = conv_bn_relu(2*in_channels, self.n_features, 3, act='relu')
    self.cbr_2 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')
    self.cbr_3 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')
    self.cbr_4 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')
    self.cbr_5 = conv_bn_relu(self.n_features, 3, 3, act=None)

  def forward(self,x):
    x = self.ap(x)
    x = self.cbr_1(x)
    x = self.cbr_2(x)
    x = self.cbr_3(x)
    x = self.cbr_4(x)
    x = self.cbr_5(x)
    x = self.up(x)
    return x  

class registration_model(pl.LightningModule):
  def __init__(self, shape, in_channels = 1, int_steps = 7, n_epoch_scale = 100):  
    super().__init__()  
    self.shape = shape
    self.transformer = SpatialTransformer(size=shape)
    self.int_steps = int_steps
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)    
    self.similarity = MSE() #NCC() #MSE()
    self.lambda_similarity = 1 
    self.lambda_grad_flow  = 0.01 #0.01 if MSE or 0.1 if NCC (from original voxelmorph article)
    self.bidir = True

    self.n_epochs = 0 
 
    self.scale_1 = velocity_model(scale_factor=1)
    self.scale_2 = velocity_model(scale_factor=2)
    self.scale_4 = velocity_model(scale_factor=4)
    self.scale_8 = velocity_model(scale_factor=8)

    self.epochs_scale8 = n_epoch_scale
    self.epochs_scale4 = n_epoch_scale*2 + self.epochs_scale8
    self.epochs_scale2 = n_epoch_scale*4 + self.epochs_scale4
    self.epochs_scale1 = n_epoch_scale*8 + self.epochs_scale2


    #self.unet_s0 = Unet(n_channels = 2, n_classes = 3, n_features = 16)
    #self.unet_s1 = Unet(n_channels = 2, n_classes = 3, n_features = 16)
    '''
    self.smoothing = GaussianSmoothing(channels=3, kernel_size=5,sigma=0.5, dim=3)

    scale_factor = 4
    self.ap = nn.AvgPool3d(scale_factor)
    self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)

    self.n_features = 32
    self.feature_1 = conv_bn_relu(2*in_channels, self.n_features, 3, act='relu')
    self.feature_2 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')
    self.feature_3 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')

    self.feature_4 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')
    self.feature_5 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')

    self.decode_2 = conv_bn_relu(self.n_features, self.n_features, 3, act='relu')
    self.decode_3 = conv_bn_relu(self.n_features, 3, 3, act=None)

    self.decode_4 = conv_bn_relu(self.n_features*2, self.n_features, 1, act='relu')
    self.decode_5 = conv_bn_relu(self.n_features, self.n_features, 3, act=None)
    self.decode_6 = conv_bn_relu(self.n_features, 3, 3, act=None)
    '''

  def on_train_epoch_end(self):
    self.n_epochs += 1

  def forward(self,source,target):
    y_source = []
    y_target = []

    forward_flow = []
    forward_velocity = []
 
    x_0 = torch.cat([source,target],dim=1)

    if self.n_epochs > self.epochs_scale2:
      forward_velocity_s = self.scale_1(x_0) + self.scale_2(x_0).detach() + self.scale_4(x_0).detach() + self.scale_8(x_0).detach()
    elif self.n_epochs > self.epochs_scale4:
      forward_velocity_s = self.scale_2(x_0) + self.scale_4(x_0).detach() + self.scale_8(x_0).detach()
    elif self.n_epochs > self.epochs_scale8:
      forward_velocity_s = self.scale_4(x_0) + self.scale_8(x_0).detach()
    else:
      forward_velocity_s = self.scale_8(x_0)

    forward_flow_s = self.vecint(forward_velocity_s)
    y_source.append(self.transformer(source, forward_flow_s))
    forward_flow.append(forward_flow_s)
    if self.bidir is True:
      backward_flow_s = self.vecint(-forward_velocity_s)
      y_target.append(self.transformer(target, backward_flow_s))

    return y_source, y_target, forward_velocity, forward_flow

  def forward_allscales(self,source,target):
    y_source = []
    y_target = []

    forward_flow = []
    forward_velocity = []

    x_0 = torch.cat([source,target],dim=1)

    fv_s8 = self.scale_8(x_0)
    fv_s4 = self.scale_4(x_0) + fv_s8
    fv_s2 = self.scale_2(x_0) + fv_s4
    fv_s1 = self.scale_1(x_0) + fv_s2
    
    forward_flow_s8 = self.vecint(fv_s8)
    backward_flow_s8 = self.vecint(-fv_s8)
    y_source.append(self.transformer(source, forward_flow_s8))
    y_target.append(self.transformer(target, backward_flow_s8))
    forward_flow.append(forward_flow_s8)

    forward_flow_s4 = self.vecint(fv_s4)
    backward_flow_s4 = self.vecint(-fv_s4)
    y_source.append(self.transformer(source, forward_flow_s4))
    y_target.append(self.transformer(target, backward_flow_s4))
    forward_flow.append(forward_flow_s4)

    forward_flow_s2 = self.vecint(fv_s2)
    backward_flow_s2 = self.vecint(-fv_s2)
    y_source.append(self.transformer(source, forward_flow_s2))
    y_target.append(self.transformer(target, backward_flow_s2))
    forward_flow.append(forward_flow_s2)

    forward_flow_s1 = self.vecint(fv_s1)
    backward_flow_s1 = self.vecint(-fv_s1)
    y_source.append(self.transformer(source, forward_flow_s1))
    y_target.append(self.transformer(target, backward_flow_s1))
    forward_flow.append(forward_flow_s1)

    '''
    #feature extraction at scale 0
    f_0 = self.feature_1(x_0)
    f_0 = self.feature_2(f_0)
    f_0 = f_0 + self.feature_3(f_0)
    
    #feature extraction at scale 1
    f_1 = self.ap(f_0)
    f_1 = self.feature_4(f_1)
    f_1 = f_1 + self.feature_5(f_1)

    fv_1 = self.up(f_1)
    fv_1 = fv_1 + self.decode_2(fv_1)
    fv_1 = self.decode_3(fv_1)
    fv_1 = self.smoothing(fv_1)


    forward_flow_s1 = self.vecint(fv_1)
    y_source_s1 = self.transformer(source, forward_flow_s1)

    backward_flow_s1 = self.vecint(-fv_1)
    y_target_s1 = self.transformer(target, backward_flow_s1)

    y_source.append(y_source_s1)
    y_target.append(y_target_s1)
    forward_flow.append(forward_flow_s1)

    fv_0 = torch.cat([f_0,self.up(f_1)],dim=1)
    fv_0 = self.decode_4(fv_0)
    fv_0 = fv_0 + self.decode_5(fv_0)
    fv_0 = self.decode_6(fv_0)

    forward_velocity_s0 = fv_0 + fv_1
    forward_flow_s0 = self.vecint(forward_velocity_s0)
    y_source_s0 = self.transformer(source, forward_flow_s0)

    backward_flow_s0 = self.vecint(-forward_velocity_s0)
    y_target_s0 = self.transformer(target, backward_flow_s0)

    y_source.append(y_source_s0)
    y_target.append(y_target_s0)
    forward_flow.append(forward_flow_s0)
    '''

    return y_source, y_target, forward_velocity, forward_flow


  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source, target = batch
    y_source, y_target, forward_velocity, forward_flow  = self(source,target)

    loss = 0 
    scale_loss = 1.0 / len(y_source) 
    for i in range(len(y_source)):
      if self.bidir is True:
        loss += scale_loss * self.lambda_similarity * (self.similarity.forward(target,y_source[i]) + self.similarity.forward(y_target[i],source))/2
      else:
        loss += scale_loss * self.lambda_similarity * self.similarity.forward(target,y_source[i])

      if self.lambda_grad_flow > 0:
        loss += self.lambda_grad_flow * Grad3d().forward(forward_flow[i]) 

    return loss 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo 3D registration')
  parser.add_argument('-s', '--source', help='Source image (nifti)', type=str, required=True)
  parser.add_argument('-t', '--target', help='Target image (nifti)', type=str, required=True)
  parser.add_argument('-w', '--warped', help='Prefix of warped (source) image (nifti)', type=str, required=True)
  parser.add_argument('-f', '--forward_flow', help='Prefix of forward flow image (nifti)', type=str, required=False)
  parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required=False, default = 10)  

  args = parser.parse_args()

  source_file = args.source
  target_file = args.target

  target_image = nibabel.load(target_file)
  source_image = nibabel.load(source_file)

  target_data  = torch.Tensor(target_image.get_fdata())
  source_data  = torch.Tensor(source_image.get_fdata())

  target_data /= torch.max(target_data)
  source_data /= torch.max(source_data)

  print('target : '+str(torch.min(target_data))+' '+str(torch.max(target_data)))
  print('source : '+str(torch.min(source_data))+' '+str(torch.max(source_data)))

  print(target_data.shape)
  print(source_data.shape)

  if len(target_data.shape) == 3:
    target_data = torch.unsqueeze(torch.unsqueeze(target_data, 0),0)
    source_data = torch.unsqueeze(torch.unsqueeze(source_data, 0),0)

  else: #4D: first permute axis and then add one dimension
    target_data = torch.unsqueeze(torch.permute(target_data, (3,0,1,2)),0)
    source_data = torch.unsqueeze(torch.permute(source_data, (3,0,1,2)),0)

  print(target_data.shape)
  x_train = torch.cat([source_data,target_data],dim=0)

  batch_size_reg = 1

  trainset_reg = TwoDataSet(x_train)
  trainloader_reg = torch.utils.data.DataLoader(trainset_reg, batch_size=batch_size_reg)   
  
  in_shape = (x_train.shape[2],x_train.shape[3],x_train.shape[4])
  reg_net = registration_model(shape=in_shape, in_channels=x_train.shape[1], n_epoch_scale=args.epochs)

  max_epochs = reg_net.epochs_scale1
  print('max_epochs:'+str(max_epochs))

  trainer_reg = pl.Trainer(gpus=1, max_epochs=max_epochs, logger=False, precision=16, enable_checkpointing=False)
  trainer_reg.fit(reg_net, trainloader_reg)  

  y_source, y_target, forward_velocity, forward_flow  = reg_net.forward_allscales(source_data,target_data)

  for i in range(len(y_source)):
    y_source_output = y_source[i]
    o = tio.ScalarImage(tensor=y_source_output[0].detach().numpy(), affine=source_image.affine)
    o.save(args.warped+'_scale_'+str(i)+'.nii.gz')

    y_target_output = y_target[i]
    o = tio.ScalarImage(tensor=y_target_output[0].detach().numpy(), affine=source_image.affine)
    o.save(args.warped+'_backward_scale_'+str(i)+'.nii.gz')

    if args.forward_flow is not None:
      forward_flow_output = forward_flow[i]
      o = tio.ScalarImage(tensor=forward_flow_output[0].detach().numpy(), affine=source_image.affine)
      o.save(args.forward_flow+'_scale_'+str(i)+'.nii.gz')
