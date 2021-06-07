#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:43:26 2021

@author: rousseau
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.modules.activation import Sigmoid
import torchio as tio
import monai


class Unet(pl.LightningModule):
    def __init__(self, n_channels = 1, n_classes = 10, n_features = 32):
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
            )


        self.dc1 = double_conv(self.n_channels, self.n_features)
        self.dc2 = double_conv(self.n_features, self.n_features*2)
        self.dc3 = double_conv(self.n_features*2, self.n_features*4)
        self.dc4 = double_conv(self.n_features*6, self.n_features*2)
        self.dc5 = double_conv(self.n_features*3, self.n_features)

        self.mp = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.dc1(x)

        x2 = self.mp(x1)
        x2 = self.dc2(x2)

        x3 = self.mp(x2)
        x3 = self.dc3(x3)

        x4 = self.up(x3)
        x4 = torch.cat([x4,x2], dim=1)
        x4 = self.dc4(x4)

        x5 = self.up(x4)
        x5 = torch.cat([x5,x1], dim=1)
        x5 = self.dc5(x5)
        return self.out(x5)

    def training_step(self, batch, batch_idx):
        patches_batch = batch
        x = patches_batch['t2'][tio.DATA]
        y = patches_batch['label'][tio.DATA]
        y_hat = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log('train_loss', loss)
        return loss        

    def validation_step(self, batch, batch_idx):
        patches_batch = batch
        x = patches_batch['t2'][tio.DATA]
        y = patches_batch['label'][tio.DATA]
        y_hat = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log('val_loss', loss)
        return loss        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Encoder(torch.nn.Module):
  def __init__(self, in_channels = 1, latent_dim = 32, n_filters = 16, patch_size = 64): 
    super(Encoder, self).__init__()
    
    #patch_size = 64
    n = 2*2*2    #3 layers of stride 2 : 64/8 * 64/8 * 64/8 * 16 -> 8192 hidden dim !
    #patch_size = 128
    n = 2*2*2*2    #4 layers of stride 2 : 128/16 * 128/16 * 128/16 * 32
    self.hidden_dim = int((patch_size / n)*(patch_size / n)*(patch_size / n)*n_filters*8)
    self.latent_dim = int(latent_dim) 

    self.enc = nn.Sequential(
      nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters, out_channels = n_filters*2, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters*2, out_channels = n_filters*4, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters*4, out_channels = n_filters*8, kernel_size = 3,stride = 2, padding=1),
      nn.Tanh(),
      nn.Flatten(),
      nn.Linear(self.hidden_dim,self.latent_dim)
      )

  def forward(self,x):
    return self.enc(x)

class Feature(torch.nn.Module):
  def __init__(self, n_channels = 1, n_features = 10, n_filters = 32):
    super(Feature, self).__init__()

    #self.unet = Unet(n_channels = n_channels, n_classes = n_features, n_features = n_filters)
    self.unet = monai.networks.nets.UNet(
                dimensions=3,
                in_channels=n_channels,
                out_channels=n_features,
                channels=(n_filters, n_filters*2, n_filters*4),
                strides=(2, 2, 2),
                num_res_units=2,
                )   

  def forward(self,x):
    xout = self.unet(x)
    #return nn.ReLU()(xout)
    return nn.Tanh()(xout)    

class Reconstruction(torch.nn.Module):
  def __init__(self, in_channels, n_filters = 16):
    super(Reconstruction, self).__init__()
    
    # self.recon = nn.Sequential(
    #   nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
    #   nn.ReLU(),
    #   nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
    #   nn.ReLU(),
    #   nn.Conv3d(in_channels = n_filters, out_channels = 1, kernel_size = 3,stride = 1, padding=1)
    #   )    
    self.recon = monai.networks.nets.UNet(
                dimensions=3,
                in_channels=in_channels,
                out_channels=1,
                channels=(n_filters, n_filters*2, n_filters*4),
                strides=(2, 2, 2),
                num_res_units=2,
                )

  def forward(self,x):
    return self.recon(x)    

class Feature2Segmentation(torch.nn.Module):
  def __init__(self, in_channels, out_channels, n_filters = 16):
    super(Feature2Segmentation, self).__init__()
    
    # self.seg = nn.Sequential(
    #   nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1,stride = 1, padding=0),
    #   #nn.Sigmoid()
    #   )  
    self.n_filters = n_filters

    self.seg = monai.networks.nets.UNet(
                dimensions=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=(self.n_filters, self.n_filters*2, self.n_filters*4),
                strides=(2, 2, 2),
                num_res_units=2,
                )

  def forward(self,x):
    return self.seg(x)        

class DecompNet(pl.LightningModule):

  def __init__(self, latent_dim = 32, n_filters = 16, patch_size = 64):
    super().__init__()
    self.patch_size = patch_size
    self.latent_dim = int(latent_dim) 
    self.n_classes = 10

    self.n_filters_encoder = int(n_filters/2)
    self.n_filters_feature = n_filters
    self.n_features = 16
    self.n_filters_recon = n_filters
    self.n_filters_seg = n_filters

    self.encoder = Encoder(self.n_features+1, latent_dim, self.n_filters_encoder, patch_size)
    self.feature = Feature(1, self.n_features, self.n_filters_feature)
    self.reconstruction = Reconstruction(self.n_features+self.latent_dim, self.n_filters_recon)
    self.segmenter = Feature2Segmentation(self.n_features, self.n_classes, self.n_filters_seg)

    self.lw = {}
    self.lw['rx'] = 1 #reconstruction-based loss
    self.lw['sx'] = 1 #segmentation loss    
    self.lw['cx'] = 0 #cross-reconstruction loss
    self.lw['ry'] = 1 #reconstruction-based loss
    self.lw['sy'] = 1 #segmentation loss   
    self.lw['cy'] = 0 #cross-reconstruction loss


  def forward(self,x,y):     
    fx = self.feature(x)

    xfx = torch.cat([x,fx], dim=1)
    zx = self.encoder(xfx)

    zx = zx.view(-1,self.latent_dim,1,1,1)
    zfx = zx.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fxzfx = torch.cat([fx,zfx], dim=1)
    rx = self.reconstruction(fxzfx)

    fy = self.feature(y)

    yfy = torch.cat([y,fy], dim=1)
    zy = self.encoder(yfy)

    zy = zy.view(-1,self.latent_dim,1,1,1)    
    zfy = zy.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fyzfy = torch.cat([fy,zfy], dim=1)
    ry = self.reconstruction(fyzfy)

    # #reconstruction of x using fy and zfx
    fyzfx = torch.cat([fy,zfx], dim=1)
    cx = self.reconstruction(fyzfx)

    # #reconstruction of y using fx and zfy
    fxzfy = torch.cat([fx,zfy], dim=1)
    cy = self.reconstruction(fxzfy)

    # #Add cycle consistency ?

    # #Segmentation
    sx = self.segmenter(fx)
    sy = self.segmenter(fy)
    
    return rx, ry, cx, cy, fx, fy, sx, sy

  def training_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    s = patches_batch['label'][tio.DATA]
    
    rx, ry, cx, cy, fx, fy, sx, sy = self(x,y)

    bce = nn.BCEWithLogitsLoss()
    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += F.mse_loss(ry, y)
      if k == 'cx' and self.lw[k] > 0:
        loss += F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += F.mse_loss(cy, y)
      if k == 'sx' and self.lw[k] > 0:
        loss += bce(sx, s.float())
      if k == 'sy' and self.lw[k] > 0:
        loss += bce(sy, s.float())

    #loss = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y) + F.mse_loss(rx, x) + F.mse_loss(ry, y) + F.mse_loss(fx, fy) + F.mse_loss(sx, s.float()) + F.mse_loss(sy, s.float())
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    s = patches_batch['label'][tio.DATA] 
    
    bce = nn.BCEWithLogitsLoss()
    x_hat, y_hat, rx, ry, fx, fy, sx, sy = self(x,y)
    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += F.mse_loss(ry, y)
      if k == 'cx' and self.lw[k] > 0:
        loss += F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += F.mse_loss(cy, y)
      if k == 'sx' and self.lw[k] > 0:
        loss += bce(sx, s.float())
      if k == 'sy' and self.lw[k] > 0:
        loss += bce(sy, s.float())
    #loss = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y) + bce(sx, s.float()) + bce(sy, s.float()) + F.mse_loss(rx, x) + F.mse_loss(ry, y)        

    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
