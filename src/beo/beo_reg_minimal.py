#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl

import torchio as tio
import numpy as np

from beo_metrics import NCC
from beo_svf import SpatialTransformer, VecInt

# Net modules
class Unet(nn.Module):
  def __init__(self, n_channels = 2, n_classes = 3, n_features = 8):
    super(Unet, self).__init__()

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.n_features = n_features

    def double_conv(in_channels, out_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
      )

    self.dc1 = double_conv(self.n_channels, self.n_features)
    self.dc2 = double_conv(self.n_features, self.n_features)
    self.dc3 = double_conv(self.n_features, self.n_features)
    self.dc4 = double_conv(self.n_features*2, self.n_features)
    self.dc5 = double_conv(self.n_features*2, self.n_features)
    
    self.ap = nn.AvgPool3d(2)

    self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    self.x3_out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)
    self.x4_out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

    self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

  def forward(self, x):
    x1 = self.dc1(x)

    x2 = self.ap(x1)
    x2 = self.dc2(x2)

    x3 = self.ap(x2)
    x3 = self.dc3(x3)

    x4 = self.up(x3)
    x4 = torch.cat([x4,x2], dim=1)
    x4 = self.dc4(x4)

    x5 = self.up(x4)
    x5 = torch.cat([x5,x1], dim=1)
    x5 = self.dc5(x5)
    return self.out(x5)



#%% Lightning module
class meta_registration_model(pl.LightningModule):
    def __init__(self, shape, int_steps = 7):  
        super().__init__()  
        self.shape = shape
        #self.unet = monai.networks.nets.Unet(spatial_dims=3, in_channels=2, out_channels=3, channels=(8,16,32), strides=(2,2))
        self.unet = Unet(n_channels = 2, n_classes = 3, n_features = 32)
        self.transformer = SpatialTransformer(size=shape)
        self.int_steps = int_steps
        self.vecint = VecInt(inshape=shape, nsteps=int_steps)
        self.loss = NCC()
        #self.loss = nn.MSELoss()
        #self.loss = monai.losses.LocalNormalizedCrossCorrelationLoss()

    def forward(self,source,target):
        x = torch.cat([source,target], dim=1)
        forward_velocity = self.unet(x)
        forward_flow = self.vecint(forward_velocity)
        warped_source = self.transformer(source, forward_flow)

        backward_velocity = - forward_velocity
        backward_flow = self.vecint(backward_velocity)
        warped_target = self.transformer(target, backward_flow)

        return warped_source, warped_target

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def training_step(self, batch, batch_idx):

        target = batch['target'][tio.DATA]
        source = batch['source'][tio.DATA]
        warped_source, warped_target = self(source,target)
        return self.loss(target,warped_source) + self.loss(source,warped_target)


#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-o', '--output', help='Output filename', type=str, required = True)

    args = parser.parse_args()

    subjects = []
    subject = tio.Subject(
        target=tio.ScalarImage(args.target),
        source=tio.ScalarImage(args.source),
    )
    subjects.append(subject) 

    normalization = tio.ZNormalization()
    #resize = tio.Resize(128)
    #transforms = [resize, normalization]
    transforms = [normalization]
    training_transform = tio.Compose(transforms)

    training_set = tio.SubjectsDataset(subjects, transform=training_transform)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=1)
#%%
    # get the spatial dimension of the data (3D)
    #in_shape = resize(subjects[0]).target.shape[1:] 
    in_shape = subjects[0].target.shape[1:]     
    reg_net = meta_registration_model(shape=in_shape)


    trainer_reg = pl.Trainer(max_epochs=args.epochs, logger=False, enable_checkpointing=False)   
    trainer_reg.fit(reg_net, training_loader)  

#%%
    # Inference
    inference_subject = training_transform(subject)
    source_data = torch.unsqueeze(inference_subject.source.data,0)
    target_data = torch.unsqueeze(inference_subject.target.data,0)    
    warped_source,warped_target = reg_net.forward(source_data,target_data)
    o = tio.ScalarImage(tensor=warped_source[0].detach().numpy(), affine=inference_subject.source.affine)
    o.save(args.output)

    #o = tio.ScalarImage(tensor=inference_subject.source.data.detach().numpy(), affine=inference_subject.source.affine)
    #o.save('source.nii.gz')
    #o = tio.ScalarImage(tensor=inference_subject.target.data.detach().numpy(), affine=inference_subject.target.affine)
    #o.save('target.nii.gz')    
