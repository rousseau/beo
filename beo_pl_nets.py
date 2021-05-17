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
        y = patches_batch['seg'][tio.DATA]
        y_hat = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log('train_loss', loss)
        return loss        

    def validation_step(self, batch, batch_idx):
        patches_batch = batch
        x = patches_batch['t2'][tio.DATA]
        y = patches_batch['seg'][tio.DATA]
        y_hat = self(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat,y)
        self.log('val_loss', loss)
        return loss        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class Encoder(torch.nn.Module):
  def __init__(self, latent_dim = 32, n_filters = 16, patch_size = 64): 
    super(Encoder, self).__init__()
    
    #patch_size = 64
    n = 2*2*2    #3 layers of stride 2 : 64/8 * 64/8 * 64/8 * 16 -> 8192 hidden dim !
    #patch_size = 128
    n = 2*2*2*2    #4 layers of stride 2 : 128/16 * 128/16 * 128/16 * 32
    self.hidden_dim = int((patch_size / n)*(patch_size / n)*(patch_size / n)*n_filters)
    self.latent_dim = int(latent_dim) 

    self.enc = nn.Sequential(
      nn.Conv3d(in_channels = 1, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.Tanh(),
      nn.Flatten(),
      nn.Linear(self.hidden_dim,self.latent_dim)
      )

  def forward(self,x):
    return self.enc(x)

class Feature(torch.nn.Module):
  def __init__(self, n_channels = 1, n_features = 10, n_filters = 32):
    super(Feature, self).__init__()

    self.unet = Unet(n_channels = n_channels, n_classes = n_features, n_features = n_filters)

  def forward(self,x):
    xout = self.unet(x)
    return nn.Tanh()(xout)
    

class Reconstruction(torch.nn.Module):
  def __init__(self, in_channels, n_filters = 16):
    super(Reconstruction, self).__init__()
    
    self.recon = nn.Sequential(
      nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
      nn.ReLU(),
      nn.Conv3d(in_channels = n_filters, out_channels = 1, kernel_size = 3,stride = 1, padding=1)
      )      
  def forward(self,x):
    return self.recon(x)    

class Feature2Segmentation(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Feature2Segmentation, self).__init__()
    
    self.seg = nn.Sequential(
      nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1,stride = 1, padding=0),
      nn.Sigmoid()
      )      
  def forward(self,x):
    return self.seg(x)        

class DecompNet(pl.LightningModule):

  def __init__(self, latent_dim = 32, n_filters = 16, patch_size = 64):
    super().__init__()
    self.patch_size = patch_size
    self.latent_dim = int(latent_dim) 
    self.n_classes = 10

    self.encoder = Encoder(latent_dim, n_filters, patch_size)
    self.feature = Feature(1, n_filters, n_filters)
    self.reconstruction = Reconstruction(n_filters+self.latent_dim, n_filters)
    self.segmenter = Feature2Segmentation(n_filters, self.n_classes)

  def forward(self,x,y): 

    fy = self.feature(y)
    sy = self.segmenter(fy)
    fx = self.feature(x)
    sx = self.segmenter(fx)
    x_hat = x
    y_hat = y
    rx = x
    ry = y
    
    # zx = self.encoder(x)
    # fx = self.feature(x)
    # zx = zx.view(-1,self.latent_dim,1,1,1)
    # zfx = zx.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    # fxzfx = torch.cat([fx,zfx], dim=1)
    # x_hat = self.reconstruction(fxzfx)
    
    # zy = self.encoder(y)
    # fy = self.feature(y)
    # zy = zy.view(-1,self.latent_dim,1,1,1)    
    # zfy = zy.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    # fyzfy = torch.cat([fy,zfy], dim=1)
    # y_hat = self.reconstruction(fyzfy)

    # #reconstruction of x using fy and zfx
    # fyzfx = torch.cat([fy,zfx], dim=1)
    # rx = self.reconstruction(fyzfx)

    # #reconstruction of y using fx and zfy
    # fxzfy = torch.cat([fx,zfy], dim=1)
    # ry = self.reconstruction(fxzfy)

    # #Add cycle consistency ?

    # #Segmentation
    # sx = self.segmenter(fx)
    # sy = self.segmenter(fy)
    
    return x_hat, y_hat, rx, ry, fx, fy, sx, sy

  def training_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    s = patches_batch['seg'][tio.DATA]
    
    x_hat, y_hat, rx, ry, fx, fy, sx, sy = self(x,y)

    #loss = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y) + F.mse_loss(rx, x) + F.mse_loss(ry, y) + F.mse_loss(fx, fy) + F.mse_loss(sx, s.float()) + F.mse_loss(sy, s.float())
    loss = F.mse_loss(sx, s.float()) + F.mse_loss(sy, s.float())
    
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    s = patches_batch['seg'][tio.DATA]
    
    x_hat, y_hat, rx, ry, fx, fy, sx, sy = self(x,y)
    loss = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y) + F.mse_loss(rx, x) + F.mse_loss(ry, y) + F.mse_loss(fx, fy) + F.mse_loss(sx, s) + F.mse_loss(sy, s)
    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
