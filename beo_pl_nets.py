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

def total_variation(x):
  return torch.sum(torch.abs(x[:, :, :, :, :-1] - x[:, :, :, :, 1:])) + \
    torch.sum(torch.abs(x[:, :, :, :-1, :] - x[:, :, :, 1:, :])) +\
    torch.sum(torch.abs(x[:, :, :-1, :, :] - x[:, :, 1:, :, :]))      


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

    # def training_step(self, batch, batch_idx):
    #     patches_batch = batch
    #     x = patches_batch['t2'][tio.DATA]
    #     y = patches_batch['label'][tio.DATA]
    #     y_hat = self(x)

    #     criterion = nn.BCEWithLogitsLoss()
    #     loss = criterion(y_hat,y)
    #     self.log('train_loss', loss)
    #     return loss        

    # def validation_step(self, batch, batch_idx):
    #     patches_batch = batch
    #     x = patches_batch['t2'][tio.DATA]
    #     y = patches_batch['label'][tio.DATA]
    #     y_hat = self(x)

    #     criterion = nn.BCEWithLogitsLoss()
    #     loss = criterion(y_hat,y)
    #     self.log('val_loss', loss)
    #     return loss        

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=1e-3)


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
      nn.Linear(self.hidden_dim,self.latent_dim),
      nn.Tanh() 
      )

  def forward(self,x):
    return self.enc(x)

class Feature(torch.nn.Module):
  def __init__(self, n_channels = 1, n_features = 10, n_filters = 32):
    super(Feature, self).__init__()

    self.unet = Unet(n_channels = n_channels, n_classes = n_features, n_features = n_filters)
    # self.unet = monai.networks.nets.UNet(
    #             dimensions=3,
    #             in_channels=n_channels,
    #             out_channels=n_features,
    #             channels=(n_filters, n_filters*2, n_filters*4),
    #             strides=(2, 2, 2),
    #             num_res_units=2,
    #             )   
                
    # self.unet = monai.networks.nets.RegUNet(
    #             spatial_dims=3,
    #             in_channels=n_channels,
    #             out_channels=n_features,
    #             num_channel_initial = n_filters, 
    #             depth = 2,
    #             pooling = True,
    #             concat_skip = True,
    #             )   


  def forward(self,x):
    xout = self.unet(x) 
    #return nn.ReLU()(xout)
    #return nn.Tanh()(xout)
    return nn.Softmax(dim=1)(xout)
    #return nn.functional.gumbel_softmax(xout, hard=True)    

class Reconstruction(torch.nn.Module):
  def __init__(self, in_channels, n_filters = 16):
    super(Reconstruction, self).__init__()
    
    ks = 3
    pad = 1
    self.recon = nn.Sequential(
       nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = 1, kernel_size = ks,stride = 1, padding=pad)
       )    
    # self.recon = monai.networks.nets.UNet(
    #             dimensions=3,
    #             in_channels=in_channels,
    #             out_channels=1,
    #             channels=(n_filters, n_filters*2, n_filters*4),
    #             strides=(2, 2, 2),
    #             num_res_units=2,
    #             )

  def forward(self,x):
    return self.recon(x)    

class Feature2Segmentation(torch.nn.Module):
  def __init__(self, in_channels, out_channels, n_filters = 16):
    super(Feature2Segmentation, self).__init__()
    
    self.n_filters = n_filters

    self.seg = nn.Sequential(
      nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
       nn.ReLU(),
       nn.Conv3d(in_channels = n_filters, out_channels = out_channels, kernel_size = 3,stride = 1, padding=1)
      #nn.Sigmoid()
      )  


    # self.seg = monai.networks.nets.UNet(
    #             dimensions=3,
    #             in_channels=in_channels,
    #             out_channels=out_channels,
    #             channels=(self.n_filters, self.n_filters*2, self.n_filters*4),
    #             strides=(2, 2, 2),
    #             num_res_units=2,
    #             )

  def forward(self,x):
    return self.seg(x)        

class Block1D(torch.nn.Module):
  def __init__(self, in_channels, n_filters = 10):  
    super(Block1D, self).__init__()
    self.n_filters = n_filters

    self.dd1 = nn.Linear(in_features = in_channels, out_features= n_filters, bias = True)
    self.dd2 = nn.Linear(in_features = n_filters, out_features= in_channels, bias = True)
    self.tanh = torch.nn.Tanh()
    self.relu = torch.nn.ReLU()

  def forward(self,x):
    x = self.dd1(x)
    x = self.relu(x)
    x = self.dd2(x)
    x = self.tanh(x)
    return x
 
class ResNet1D_forward(torch.nn.Module):
  def __init__(self, block, n_layers = 10):  
    super(ResNet1D_forward, self).__init__()
    self.n_layers = n_layers
    self.block = block

  def forward(self,x):
    for i in range(self.n_layers):
      y = self.block(x)
      x = torch.add(x,y)
    return x  

class ResNet1D_backward(torch.nn.Module):
  def __init__(self, block, n_layers = 10, order = 1):  
    super(ResNet1D_backward, self).__init__()
    self.n_layers = n_layers
    self.order = order
    self.block = block

  def forward(self,x):
    z = x
    for i in range(self.order): #fixed point iterations
      y = self.block(z)
      z = torch.sub(x,y)
    x = z    
    return x  

class DecompNet(pl.LightningModule):

  def __init__(self, latent_dim = 32, n_filters = 16, n_features = 16, patch_size = 64, learning_rate = 1e-4):
    super().__init__()
    self.patch_size = patch_size
    self.latent_dim = int(latent_dim) 
    self.n_classes = 10

    self.learning_rate = learning_rate

    self.n_filters_encoder = int(n_filters/2)
    self.n_filters_feature = n_filters
    self.n_features = n_features
    self.n_filters_recon = n_filters
    self.n_filters_seg = n_filters

    self.encoder = Encoder(self.n_features+1, latent_dim, self.n_filters_encoder, patch_size)
    self.feature_x = Feature(1, self.n_features, self.n_filters_feature)
    self.feature_y = Feature(1, self.n_features, self.n_filters_feature)    
    self.reconstruction = Reconstruction(self.n_features+self.latent_dim, self.n_filters_recon)
    self.segmenter = Feature2Segmentation(self.n_features, self.n_classes, self.n_filters_seg)

    self.lw = {}
    self.lw['rx'] = 1 #reconstruction-based loss
    self.lw['sx'] = 1 #segmentation loss    
    self.lw['cx'] = 1 #cross-reconstruction loss
    self.lw['ry'] = 1 #reconstruction-based loss
    self.lw['sy'] = 1 #segmentation loss   
    self.lw['cy'] = 1 #cross-reconstruction loss
    self.lw['tvx'] = 0 #total variation loss
    self.lw['tvy'] = 0 #total variation loss
    self.lw['fxfy'] = 0 #shared feature-based loss
    self.lw['varxy'] = 1 #variance of the latent vector

    for k in self.lw.keys():
      print(k+' : '+str(self.lw[k]))

  def forward(self,x,y):     
    fx = self.feature_x(x)

    xfx = torch.cat([x,fx], dim=1)
    zx = self.encoder(xfx)

    zx = zx.view(-1,self.latent_dim,1,1,1)
    zfx = zx.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fxzfx = torch.cat([fx,zfx], dim=1)
    rx = self.reconstruction(fxzfx)

    fy = self.feature_y(y)

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
    
    return rx, ry, cx, cy, fx, fy, sx, sy, zx, zy

  def training_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    s = patches_batch['label'][tio.DATA]
    
    rx, ry, cx, cy, fx, fy, sx, sy, zx, zy = self(x,y)

    bce = nn.BCEWithLogitsLoss()
    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(ry, y)
      if k == 'cx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cy, y)
      if k == 'sx' and self.lw[k] > 0:
        loss += self.lw[k] * bce(sx, s.float())
      if k == 'sy' and self.lw[k] > 0:
        loss += self.lw[k] * bce(sy, s.float())
      if k == 'tvx' and self.lw[k] > 0:
        loss += self.lw[k] * total_variation(fx)
      if k == 'tvy' and self.lw[k] > 0:
        loss += self.lw[k] * total_variation(fy)
      if k == 'fxfy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(fx, fy) 
      if k == 'varxy' and self.lw[k] > 0:
        loss += self.lw[k] * (torch.mean(torch.var(zx,dim=1)) + torch.mean(torch.var(zy,dim=1))) 

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    s = patches_batch['label'][tio.DATA] 
    
    bce = nn.BCEWithLogitsLoss()
    rx, ry, cx, cy, fx, fy, sx, sy, zx, zy = self(x,y)
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
      if k == 'tvx' and self.lw[k] > 0:
        loss += self.lw[k] * total_variation(fx)
      if k == 'tvy' and self.lw[k] > 0:
        loss += self.lw[k] * total_variation(fy)
      if k == 'fxfy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(fx, fy) 
      if k == 'varxy' and self.lw[k] > 0:
        loss += self.lw[k] * (torch.mean(torch.var(zx,dim=1)) + torch.mean(torch.var(zy,dim=1))) 

    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer

class DecompNet_IXI(pl.LightningModule):

  def __init__(self, latent_dim = 32, n_filters_encoder = 16, n_filters_feature = 16, n_filters_recon = 16, n_features = 16, patch_size = 64, learning_rate = 1e-4):
    super().__init__()
    self.patch_size = patch_size
    self.latent_dim = int(latent_dim) 
    self.n_classes = 10

    self.learning_rate = learning_rate

    self.n_filters_encoder = n_filters_encoder
    self.n_filters_feature = n_filters_feature
    self.n_features = n_features
    self.n_filters_recon = n_filters_recon

    self.encoder = Encoder(self.n_features+1, latent_dim, self.n_filters_encoder, patch_size)
    self.feature_x = Feature(1, self.n_features, self.n_filters_feature)
    self.feature_y = Feature(1, self.n_features, self.n_filters_feature)    
    self.feature_z = Feature(1, self.n_features, self.n_filters_feature)    
    self.reconstruction = Reconstruction(self.n_features+self.latent_dim, self.n_filters_recon)

    self.n_layers = 10
    self.block_x_to_y = Block1D(in_channels = self.latent_dim, n_filters = self.latent_dim * 2)
    self.mapping_x_to_y = ResNet1D_forward(block = self.block_x_to_y, n_layers = self.n_layers)
    self.mapping_y_to_x = ResNet1D_backward(block = self.block_x_to_y, n_layers = self.n_layers)    

    self.block_y_to_z = Block1D(in_channels = self.latent_dim, n_filters = self.latent_dim * 2)
    self.mapping_y_to_z = ResNet1D_forward(block = self.block_y_to_z, n_layers = self.n_layers)
    self.mapping_z_to_y = ResNet1D_backward(block = self.block_y_to_z, n_layers = self.n_layers)    

    self.block_z_to_x = Block1D(in_channels = self.latent_dim, n_filters = self.latent_dim * 2)
    self.mapping_z_to_x = ResNet1D_forward(block = self.block_z_to_x, n_layers = self.n_layers)
    self.mapping_x_to_z = ResNet1D_backward(block = self.block_z_to_x, n_layers = self.n_layers)    

    self.lw = {}
    self.lw['rx'] = 1 #reconstruction-based loss
    self.lw['cx'] = 1 #cross-reconstruction loss
    self.lw['ry'] = 1 #reconstruction-based loss
    self.lw['cy'] = 1 #cross-reconstruction loss
    self.lw['rz'] = 1 #reconstruction-based loss
    self.lw['cz'] = 1 #cross-reconstruction loss
    self.lw['m'] = 0  #mapping based cross-reconstruction loss

  def forward(self,x,y,z):
    #First modality---------------     
    fx = self.feature_x(x)

    xfx = torch.cat([x,fx], dim=1)
    zx = self.encoder(xfx)

    zx5d = zx.view(-1,self.latent_dim,1,1,1)
    zxf = zx5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fxzxf = torch.cat([fx,zxf], dim=1)
    rx = self.reconstruction(fxzxf)

    #Second modality---------------     
    fy = self.feature_y(y)

    yfy = torch.cat([y,fy], dim=1)
    zy = self.encoder(yfy)

    zy5d = zy.view(-1,self.latent_dim,1,1,1)    
    zyf = zy5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fyzyf = torch.cat([fy,zyf], dim=1)
    ry = self.reconstruction(fyzyf)

    #Third modality---------------     
    fz = self.feature_z(z)

    zfz = torch.cat([z,fz], dim=1)
    zz = self.encoder(zfz)

    zz5d = zz.view(-1,self.latent_dim,1,1,1)
    zzf = zz5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fzzzf = torch.cat([fz,zzf], dim=1)
    rz = self.reconstruction(fzzzf)

    #One way to do this only one time : mean of feature maps
    f = torch.mean(torch.stack([fx,fy,fz]),dim=0)

    # Reconstruction of x using f and zfx
    fzxf = torch.cat([f,zxf], dim=1)
    cx = self.reconstruction(fzxf)

    # Reconstruction of y using f and zfy
    fzyf = torch.cat([f,zyf], dim=1)
    cy = self.reconstruction(fzyf)

    # Reconstruction of z using f and zzf
    fzzf = torch.cat([f,zzf], dim=1)
    cz = self.reconstruction(fzzf)

    # #Add cycle consistency ?

    # Reconstruction using latent transfert
    # X -> Y
    zx2y = self.mapping_x_to_y(zx)
    zx2y5d = zx2y.view(-1,self.latent_dim,1,1,1)
    zx2yf = zx2y5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fx2y = torch.cat([f,zx2yf], dim=1)    
    mx2y = self.reconstruction(fx2y)

    zy2x = self.mapping_y_to_x(zy)
    zy2x5d = zy2x.view(-1,self.latent_dim,1,1,1)
    zy2xf = zy2x5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fy2x = torch.cat([f,zy2xf], dim=1)    
    my2x = self.reconstruction(fy2x)

    # Y -> Z
    zy2z = self.mapping_y_to_z(zy)
    zy2z5d = zy2z.view(-1,self.latent_dim,1,1,1)
    zy2zf = zy2z5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fy2z = torch.cat([f,zy2zf], dim=1)    
    my2z = self.reconstruction(fy2z)

    zz2y = self.mapping_z_to_y(zz)
    zz2y5d = zz2y.view(-1,self.latent_dim,1,1,1)
    zz2yf = zz2y5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fz2y = torch.cat([f,zz2yf], dim=1)    
    mz2y = self.reconstruction(fz2y)

    # Z -> X
    zz2x = self.mapping_z_to_x(zz)
    zz2x5d = zz2x.view(-1,self.latent_dim,1,1,1)
    zz2xf = zz2x5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fz2x = torch.cat([f,zz2xf], dim=1)    
    mz2x = self.reconstruction(fz2x)

    zx2z = self.mapping_x_to_z(zx)
    zx2z5d = zx2z.view(-1,self.latent_dim,1,1,1)
    zx2zf = zx2z5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fx2z = torch.cat([f,zx2zf], dim=1)    
    mx2z = self.reconstruction(fx2z)


  
    return rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y, my2z, mz2y, mz2x, mx2z

  def training_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    z = patches_batch['pd'][tio.DATA]
    
    rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y, my2z, mz2y, mz2x, mx2z = self(x,y,z)

    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(ry, y)
      if k == 'rz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rz, z)
      if k == 'cx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cy, y)
      if k == 'cz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cz, z)
      if k == 'm' and self.lw[k] > 0:
        loss += self.lw[k] * (F.mse_loss(my2x, x) + F.mse_loss(mx2y, y))
        loss += self.lw[k] * (F.mse_loss(my2z, z) + F.mse_loss(mz2y, y))
        loss += self.lw[k] * (F.mse_loss(mz2x, x) + F.mse_loss(mx2z, z))

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    z = patches_batch['pd'][tio.DATA]
    
    rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y, my2z, mz2y, mz2x, mx2z = self(x,y,z)

    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(ry, y)
      if k == 'rz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rz, z)
      if k == 'cx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cy, y)
      if k == 'cz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cz, z)
      if k == 'm' and self.lw[k] > 0:
        loss += self.lw[k] * (F.mse_loss(my2x, x) + F.mse_loss(mx2y, y))
        loss += self.lw[k] * (F.mse_loss(my2z, z) + F.mse_loss(mz2y, y))
        loss += self.lw[k] * (F.mse_loss(mz2x, x) + F.mse_loss(mx2z, z))

    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer

class DecompNet_3(pl.LightningModule):

  def __init__(self, latent_dim = 32, n_filters_encoder = 16, n_filters_feature = 16, n_filters_recon = 16, n_features = 16, patch_size = 64, learning_rate = 1e-4):
    super().__init__()
    self.patch_size = patch_size
    self.latent_dim = int(latent_dim) 
    self.n_classes = 10

    self.learning_rate = learning_rate

    self.n_filters_encoder = n_filters_encoder
    self.n_filters_feature = n_filters_feature
    self.n_features = n_features 
    self.n_filters_recon = n_filters_recon

    self.encoder = Encoder(self.n_features+1, latent_dim, self.n_filters_encoder, patch_size)
    self.feature = Feature(1, self.n_features, self.n_filters_feature)
    self.reconstruction = Reconstruction(self.n_features+self.latent_dim, self.n_filters_recon)

    self.n_layers = 10
    self.block_x_to_y = Block1D(in_channels = self.latent_dim, n_filters = self.latent_dim * 2)
    self.mapping_x_to_y = ResNet1D_forward(block = self.block_x_to_y, n_layers = self.n_layers)
    self.mapping_y_to_x = ResNet1D_backward(block = self.block_x_to_y, n_layers = self.n_layers)    

    self.block_y_to_z = Block1D(in_channels = self.latent_dim, n_filters = self.latent_dim * 2)
    self.mapping_y_to_z = ResNet1D_forward(block = self.block_y_to_z, n_layers = self.n_layers)
    self.mapping_z_to_y = ResNet1D_backward(block = self.block_y_to_z, n_layers = self.n_layers)    

    self.block_z_to_x = Block1D(in_channels = self.latent_dim, n_filters = self.latent_dim * 2)
    self.mapping_z_to_x = ResNet1D_forward(block = self.block_z_to_x, n_layers = self.n_layers)
    self.mapping_x_to_z = ResNet1D_backward(block = self.block_z_to_x, n_layers = self.n_layers)    

    self.lw = {}
    self.lw['rx'] = 1 #reconstruction-based loss
    self.lw['cx'] = 1 #cross-reconstruction loss
    self.lw['ry'] = 1 #reconstruction-based loss
    self.lw['cy'] = 1 #cross-reconstruction loss
    self.lw['rz'] = 1 #reconstruction-based loss
    self.lw['cz'] = 1 #cross-reconstruction loss
    self.lw['m'] = 0  #mapping based cross-reconstruction loss
    self.lw['f'] = 1  #feature similarity loss

  def forward(self,x,y,z):
    #First modality---------------     
    fx = self.feature(x)

    xfx = torch.cat([x,fx], dim=1)
    zx = self.encoder(xfx)

    zx5d = zx.view(-1,self.latent_dim,1,1,1)
    zxf = zx5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fxzxf = torch.cat([fx,zxf], dim=1)
    rx = self.reconstruction(fxzxf)

    #Second modality---------------     
    fy = self.feature(y)

    yfy = torch.cat([y,fy], dim=1)
    zy = self.encoder(yfy)

    zy5d = zy.view(-1,self.latent_dim,1,1,1)    
    zyf = zy5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fyzyf = torch.cat([fy,zyf], dim=1)
    ry = self.reconstruction(fyzyf)

    #Third modality---------------     
    fz = self.feature(z)

    zfz = torch.cat([z,fz], dim=1)
    zz = self.encoder(zfz)

    zz5d = zz.view(-1,self.latent_dim,1,1,1)
    zzf = zz5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fzzzf = torch.cat([fz,zzf], dim=1)
    rz = self.reconstruction(fzzzf)

    #One way to do this only one time : mean of feature maps
    f = torch.mean(torch.stack([fx,fy,fz]),dim=0)

    # Reconstruction of x using f and zfx
    fzxf = torch.cat([fy,zxf], dim=1)
    cx = self.reconstruction(fzxf)

    # Reconstruction of y using f and zfy
    fzyf = torch.cat([fz,zyf], dim=1)
    cy = self.reconstruction(fzyf)
 
    # Reconstruction of z using f and zzf
    fzzf = torch.cat([fx,zzf], dim=1)
    cz = self.reconstruction(fzzf)

    # #Add cycle consistency ?

    # Reconstruction using latent transfert
    # X -> Y
    zx2y = self.mapping_x_to_y(zx)
    zx2y5d = zx2y.view(-1,self.latent_dim,1,1,1)
    zx2yf = zx2y5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fx2y = torch.cat([f,zx2yf], dim=1)    
    mx2y = self.reconstruction(fx2y)

    zy2x = self.mapping_y_to_x(zy)
    zy2x5d = zy2x.view(-1,self.latent_dim,1,1,1)
    zy2xf = zy2x5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fy2x = torch.cat([f,zy2xf], dim=1)    
    my2x = self.reconstruction(fy2x)

    # Y -> Z
    zy2z = self.mapping_y_to_z(zy)
    zy2z5d = zy2z.view(-1,self.latent_dim,1,1,1)
    zy2zf = zy2z5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fy2z = torch.cat([f,zy2zf], dim=1)    
    my2z = self.reconstruction(fy2z)

    zz2y = self.mapping_z_to_y(zz)
    zz2y5d = zz2y.view(-1,self.latent_dim,1,1,1)
    zz2yf = zz2y5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fz2y = torch.cat([f,zz2yf], dim=1)    
    mz2y = self.reconstruction(fz2y)

    # Z -> X
    zz2x = self.mapping_z_to_x(zz)
    zz2x5d = zz2x.view(-1,self.latent_dim,1,1,1)
    zz2xf = zz2x5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fz2x = torch.cat([f,zz2xf], dim=1)    
    mz2x = self.reconstruction(fz2x)

    zx2z = self.mapping_x_to_z(zx)
    zx2z5d = zx2z.view(-1,self.latent_dim,1,1,1)
    zx2zf = zx2z5d.repeat(1,1,self.patch_size,self.patch_size,self.patch_size)
    fx2z = torch.cat([f,zx2zf], dim=1)    
    mx2z = self.reconstruction(fx2z)


  
    return rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y, my2z, mz2y, mz2x, mx2z

  def training_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    z = patches_batch['pd'][tio.DATA]
    
    rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y, my2z, mz2y, mz2x, mx2z = self(x,y,z)

    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(ry, y)
      if k == 'rz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rz, z)
      if k == 'cx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cy, y)
      if k == 'cz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cz, z)
      if k == 'm' and self.lw[k] > 0:
        loss += self.lw[k] * (F.mse_loss(my2x, x) + F.mse_loss(mx2y, y))
        loss += self.lw[k] * (F.mse_loss(my2z, z) + F.mse_loss(mz2y, y))
        loss += self.lw[k] * (F.mse_loss(mz2x, x) + F.mse_loss(mx2z, z))
      if k == 'f' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(fx, fy)  
        loss += self.lw[k] * F.mse_loss(fz, fy)  
        loss += self.lw[k] * F.mse_loss(fx, fz)  

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    patches_batch = batch
    x = patches_batch['t1'][tio.DATA]
    y = patches_batch['t2'][tio.DATA]
    z = patches_batch['pd'][tio.DATA]
    
    rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y, my2z, mz2y, mz2x, mx2z = self(x,y,z)

    loss = 0
    for k in self.lw.keys():
      if k == 'rx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rx, x)
      if k == 'ry' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(ry, y)
      if k == 'rz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(rz, z)
      if k == 'cx' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cx, x)
      if k == 'cy' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cy, y)
      if k == 'cz' and self.lw[k] > 0:
        loss += self.lw[k] * F.mse_loss(cz, z)
      if k == 'm' and self.lw[k] > 0:
        loss += self.lw[k] * (F.mse_loss(my2x, x) + F.mse_loss(mx2y, y))
        loss += self.lw[k] * (F.mse_loss(my2z, z) + F.mse_loss(mz2y, y))
        loss += self.lw[k] * (F.mse_loss(mz2x, x) + F.mse_loss(mx2z, z))

    self.log('val_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer
