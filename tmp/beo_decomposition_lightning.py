#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import matplotlib.pyplot as plt

#%%
print('loading data')
import joblib 
from os.path import expanduser
home = expanduser("~")
import numpy as np

patch_size = 64 
X_train = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_training.pkl')
X_train = np.moveaxis(X_train,3,1)  
Y_train = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_training.pkl')
Y_train = np.moveaxis(Y_train,3,1)  
X_test = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_testing.pkl')
X_test = np.moveaxis(X_test,3,1)  
Y_test = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_testing.pkl')
Y_test = np.moveaxis(Y_test,3,1)  

n_training_samples = X_train.shape[0]
batch_size = 32 
trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=32)

#%%

class Encoder(torch.nn.Module):
  def __init__(self, latent_dim = 32, n_filters = 16, patch_size = 64):
    super(Encoder, self).__init__()
    
    self.hidden_dim = int((patch_size / 4)*(patch_size / 4)*n_filters)
    self.latent_dim = int(latent_dim) 

    self.enc = nn.Sequential(
      nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 2, padding=1),
      nn.Tanh(),
      nn.Flatten(),
      nn.Linear(self.hidden_dim,self.latent_dim)
      )

  def forward(self,x):
    return self.enc(x)

class Feature(torch.nn.Module):
  def __init__(self, n_filters = 16):
    super(Feature, self).__init__()
    self.feat = nn.Sequential(
      nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
      nn.Tanh()
      )
  def forward(self,x):
    return self.feat(x)

class Reconstruction(torch.nn.Module):
  def __init__(self, in_channels, n_filters = 16):
    super(Reconstruction, self).__init__()
    
    self.recon = nn.Sequential(
      nn.Conv2d(in_channels = in_channels, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels = n_filters, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels = n_filters, out_channels = 1, kernel_size = 3,stride = 1, padding=1)
      )      
  def forward(self,x):
    return self.recon(x)    

class LitAutoEncoder(pl.LightningModule):

  def __init__(self, latent_dim = 32, n_filters = 16, patch_size = 64):
    super().__init__()
    self.patch_size = patch_size
    self.hidden_dim = int((self.patch_size / 4)*(self.patch_size / 4)*n_filters)
    self.latent_dim = int(latent_dim) 

    self.encoder = Encoder(latent_dim, n_filters, patch_size)
    self.feature = Feature(n_filters)
    self.reconstruction = Reconstruction(n_filters+self.latent_dim, n_filters)
    
    # self.decoder = nn.Sequential(
    #   nn.Linear(self.latent_dim,self.hidden_dim),
    #   nn.Unflatten(1,(n_filters,int(patch_size/4),int(patch_size/4))),
    #   nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=2, padding = 1, output_padding=1),
    #   nn.ReLU(),
    #   nn.ConvTranspose2d(in_channels=n_filters, out_channels=1, kernel_size=3, stride=2, padding = 1, output_padding=1)
    #   )


    # self.latent2feature = nn.Sequential(
    #   nn.Linear(self.latent_dim,self.hidden_dim),
    #   nn.Unflatten(1,(n_filters,int(patch_size/4),int(patch_size/4))),
    #   nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=2, padding = 1, output_padding=1),
    #   nn.ReLU(),
    #   nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=2, padding = 1, output_padding=1)
    #   )


  def forward(self,x,y): 
    zx = self.encoder(x)
    fx = self.feature(x)
    #zfx= self.latent2feature(zx)
    zx = zx.view(-1,self.latent_dim,1,1)
    zfx = zx.repeat(1,1,self.patch_size,self.patch_size)
    #x_hat = self.decoder(zx)
    fxzfx = torch.cat([fx,zfx], dim=1)
    x_hat = self.reconstruction(fxzfx)
    
    zy = self.encoder(y)
    fy = self.feature(y)
    #zfy= self.latent2feature(zy)
    zy = zy.view(-1,self.latent_dim,1,1)    
    zfy = zy.repeat(1,1,self.patch_size,self.patch_size)
    #y_hat = self.decoder(zy)
    fyzfy = torch.cat([fy,zfy], dim=1)
    y_hat = self.reconstruction(fyzfy)

    #reconstruction of x using fy and zfx
    fyzfx = torch.cat([fy,zfx], dim=1)
    rx = self.reconstruction(fyzfx)

    #reconstruction of y using fx and zfy
    fxzfy = torch.cat([fx,zfy], dim=1)
    ry = self.reconstruction(fxzfy)

    #Add cycle consistency ?
    
    return x_hat, y_hat, rx, ry, fx, fy

  def training_step(self, batch, batch_idx):
    x, y = batch
    x_hat, y_hat, rx, ry, fx, fy = self(x,y)
    loss = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y) + F.mse_loss(rx, x) + F.mse_loss(ry, y) + F.mse_loss(fx, fy)
    self.log('train_loss', loss)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

#%%
latent_dim = 10
n_filters = 32
net = LitAutoEncoder(latent_dim = latent_dim, n_filters = n_filters)
trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=20)

toto = torch.randn(32, 1, 64, 64)
titi = net.encoder(toto)
print(titi.size())
tata = net.feature(toto)
print(tata.size())


print('training ...')
trainer.fit(net, trainloader)

#%%
print('visualization')
#Take 10 examples for visualization
n = 10
n_testing_samples = X_test.shape[0]
step = (int)(n_testing_samples / n)
t1_vis = torch.Tensor(X_test[0:n*step:step,:,:,:])
t2_vis = torch.Tensor(Y_test[0:n*step:step,:,:,:])

net.eval()
[a,b,c,d,e,f] = net(t1_vis,t2_vis)

print(a.size())

from beo_visu import show_patches
output_path = home+'/Sync/Experiments/HCP/'
prefix = 'ae'

show_patches(patch_list=[t1_vis.cpu().detach().numpy(),
                        t2_vis.cpu().detach().numpy(),
                        a.cpu().detach().numpy(),
                        b.cpu().detach().numpy(),
                        c.cpu().detach().numpy(),
                        d.cpu().detach().numpy()],
            titles = ['$x$','$y$','$\hat{x}$','$\hat{y}$','$rx$','$ry$'],
            filename=output_path+prefix+'_fig_patches.png')


#todo : visualiser les codes T1 et T2 avec umap
z1 = net.encoder(t1_vis)
z2 = net.encoder(t2_vis)
f1 = net.feature(t1_vis)
f2 = net.feature(t2_vis)
print(z1.size())

import numpy as np
n_interp = 10
res = []
for i in range(n):

  z = torch.stack([z1[i,:] + (z2[i,:] - z1[i,:])*t for t in np.linspace(0, 1, n_interp)])
  z = z.view(-1,latent_dim,1,1)
  z = z.repeat(1,1,patch_size,patch_size)

  f = f1[i,:,:,:].repeat(n_interp,1,1,1)
  fz = torch.cat([f,z], dim=1)
  r = net.reconstruction(fz)
  res.append(r.cpu().detach().numpy())
  
show_patches(patch_list=res,
            titles = [t for t in np.linspace(0, 1, n)],
            filename=output_path+prefix+'_interp_fig_patches.png')

