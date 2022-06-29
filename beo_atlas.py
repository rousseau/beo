#!/usr/bin/env python3
#%%
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:41:16 2022

@author: rousseau
"""
import os
from os.path import expanduser
home = expanduser("~")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from skimage.transform import resize
from skimage.io import imsave

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image
from scipy.ndimage import gaussian_filter

display_plot = False

#%%
def visu_img(imgs):
  n_imgs = len(imgs)
  fig, axs = plt.subplots(1, n_imgs)
  if n_imgs == 1:
    axs = [axs]    
  for i in range(n_imgs):
    axs[i].imshow(imgs[i], interpolation="nearest", cmap="gray")    
  plt.show()
#%%
file_npz = home+'/Sync-Exp/Data/dhcp_2d.npz'
if os.path.exists(file_npz):
  array_npz = np.load(file_npz)  
  X = array_npz['arr_0']
  Y = array_npz['arr_1']
  
else:    
  #data_path = '/Volumes/2big/Sync-Data/Cerveau/Neonatal/dhcp-rel-3/'
  data_path = '/media/rousseau/Seagate5To/Sync-Data/Cerveau/Neonatal/dhcp-rel-3/'
    
  file_tsv = pd.read_csv(data_path+'combined.tsv', sep='\t')
  print(file_tsv.head())
  
  if display_plot:
    age = np.array(file_tsv['scan_age'])
    plt.figure()
    plt.hist(age, bins=25)
    plt.show(block=False)
  
  n_images = file_tsv.shape[0]
  
  slices = []
  ages = []
  
  for i in range(n_images):
    subject = str(file_tsv['participant_id'][i])
    session = str(file_tsv['session_id'][i])
    age = file_tsv['scan_age'][i]
    
    
    file_nifti = data_path+'sub-'+subject+'/ses-'+session+'/anat/sub-'+subject+'_ses-'+session+'_desc-restore_T2w.nii.gz'
  
    if os.path.exists(file_nifti):
  
      print(file_nifti)
      im_nifti = nibabel.load(file_nifti) #size : 217 x 290 x 290
      array = im_nifti.get_fdata()
      #print(array.shape)
    
      array_2d = array[:,:,int(array.shape[2]/2)]
      resized_array = resize(array_2d, (192,256))
      
      normalized_array = (resized_array - np.mean(resized_array)) / np.std(resized_array)
      slices.append(np.expand_dims(normalized_array, axis = 0))
      ages.append(age)
        
  X = np.concatenate(slices,axis=0)
  Y = np.array(ages)    
  
  if display_plot:
    plt.figure()
    plt.imshow(X[0,:,:],cmap='gray')
    plt.show(block=False)
  
  np.savez(file_npz,X,Y)


#%%
class Unet(nn.Module):
    def __init__(self, n_channels = 2, n_classes = 2, n_features = 8):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )


        self.dc1 = double_conv(self.n_channels, self.n_features)
        self.dc2 = double_conv(self.n_features, self.n_features*2)
        self.dc3 = double_conv(self.n_features*2, self.n_features*4)
        self.dc4 = double_conv(self.n_features*6, self.n_features*2)
        self.dc5 = double_conv(self.n_features*3, self.n_features)
        self.mp = nn.MaxPool2d(2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1)

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

# code from voxelmorph repo
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)  

#from voxelmorph repo
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class Grad:
    """
    2D gradient loss.
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def loss(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        return grad      
      
#%%
class registration_model(pl.LightningModule):
  def __init__(self, shape, int_steps = 7, n_features = 8):  
    super().__init__()  
    self.shape = shape

    self.unet_model = Unet(n_features = n_features)
    self.transformer = SpatialTransformer(size=shape)
    self.int_steps = int_steps
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)

    self.lambda_similarity = 1
    self.lambda_grad_flow  = 1
    self.lambda_magn_flow  = 0.01

  def forward(self,source,target):
    #concatenate images for unet
    x = torch.cat([source,target],dim=1)
    forward_velocity = self.unet_model(x)
    
    backward_velocity = -forward_velocity
    if self.int_steps > 0:
      forward_flow = self.vecint(forward_velocity)
      backward_flow= self.vecint(backward_velocity)
    
    y_source = self.transformer(source, forward_flow)
    y_target = self.transformer(target, backward_flow)
    
    return y_source, y_target, forward_velocity, forward_flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source, target = batch

    y_source,y_target, forward_velocity, forward_flow = self(source,target)
    
    loss = self.lambda_similarity * (F.mse_loss(target,y_source) + F.mse_loss(y_target,source))/2
    loss += self.lambda_grad_flow * Grad().loss(forward_flow) 
    loss += self.lambda_magn_flow * F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)    
    
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

    #_source = torch.unsqueeze(self.X[index_source],0)
    #_target = torch.unsqueeze(self.X[index_target],0)
    _source = self.X[index_source][0]
    _target = self.X[index_target][0]
    
    return _source, _target

#%%
batch_size_reg = 32
n_training = X.shape[0]
x_train = torch.reshape(torch.Tensor(X[:n_training, ...]),(n_training,1,X.shape[1],X.shape[2]))

trainset_reg = CustomDataSet(x_train)
trainloader_reg = torch.utils.data.DataLoader(trainset_reg, batch_size=batch_size_reg)

#%%  
n_epochs_reg = 100
in_shape = (X.shape[1],X.shape[2])
n_features = 16

reg_net = registration_model(shape=in_shape, n_features = n_features)

trainer_reg = pl.Trainer(gpus=1, 
                     max_epochs=n_epochs_reg)
trainer_reg.fit(reg_net, trainloader_reg)  

trainer_reg.save_checkpoint(home+'/Sync-Exp/reg_model_'+str(n_epochs_reg)+'_nf'+str(n_features)+'.ckpt') 

#%%
#Check reg network

n_source = 1
n_target = 2

source_np = X[n_source, ...]
target_np = X[n_target, ...]

source = torch.reshape(torch.Tensor(source_np),(1,1,X.shape[1],X.shape[2]))
target =  torch.reshape(torch.Tensor(target_np),(1,1,X.shape[1],X.shape[2]))

y_source, y_target, _, _ = reg_net.forward(source,target)
y_source_np = np.reshape(y_source.cpu().detach().numpy(),(in_shape))
y_target_np = np.reshape(y_target.cpu().detach().numpy(),(in_shape))
visu_img([source_np, target_np, y_source_np,y_target_np])

#%%
class atlas_building_model(pl.LightningModule):
  def __init__(self, reg_net = reg_net, init_atlas = None):
    super().__init__()  
    self.shape = reg_net.shape

    if init_atlas is None:
      self.atlas = nn.Parameter(torch.unsqueeze(torch.randn(self.shape) / 10,0))
    else:
      self.atlas = nn.Parameter(torch.unsqueeze(torch.Tensor(init_atlas),0))

    self.reg_net = reg_net
    self.e = 0

    self.lambda_similarity = 1
    self.lambda_grad_flow  = 1
    self.lambda_magn_flow  = 0.01
    self.lambda_mean_flow  = 1
    

  def forward(self,source): # takes only one image, i.e. batch size = 1
    target = torch.unsqueeze(self.atlas,0)
    
    y_source, y_target, forward_velocity, forward_flow = self.reg_net(source, target)
    
    return y_source, y_target, forward_velocity, forward_flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source_batch, _ = batch

    loss = 0
    target = torch.unsqueeze(self.atlas,0)
    sum_flows = torch.zeros((1,2*source_batch.shape[2]*source_batch.shape[3]),device=self.device)
    n_images = torch.tensor(source_batch.shape[0],device=self.device)
    
    for i in range(n_images):
      source = torch.unsqueeze(source_batch[i],0)
      
      y_source, y_target, forward_velocity, forward_flow = self(source)
      
      loss += self.lambda_similarity * F.mse_loss(target,y_source)# + F.mse_loss(y_target,source)  #Dey does the opposite (warp the atlas onto the images)
      loss += self.lambda_grad_flow * Grad().loss(forward_flow) 
      loss += self.lambda_magn_flow * F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)
        
      sum_flows += torch.reshape(forward_flow,(1,-1)) 
      
    loss += loss / n_images + self.lambda_mean_flow * F.mse_loss(torch.zeros(sum_flows.shape,device=self.device), sum_flows / n_images) #torch.norm(sum_flows / n_images,p=2)

    return loss       
  
  def training_epoch_end(self, training_step_outputs):
    self.e += 1
    #all_preds = torch.stack(training_step_outputs)
    if (self.e-1) % 10 == 0: 
      min_val = torch.min(self.atlas[0,:,:])
      max_val = torch.max(self.atlas[0,:,:])      
      #save_image((self.atlas[0,:,:]-min_val)*255/(max_val-min_val), home+'/Sync-Exp/atlas_'+str(self.e).zfill(4)+'.png')
      imsave(home+'/Sync-Exp/atlas_'+str(self.e).zfill(4)+'.png',((self.atlas[0,:,:]-min_val)*255/(max_val-min_val)).cpu().detach().numpy().astype(np.uint8))



#%%
# Prepare data for pytorch
batch_size = 128

n_training = X.shape[0]
x_train = torch.reshape(torch.Tensor(X[:n_training, ...]),(n_training,1,X.shape[1],X.shape[2]))
y_train = torch.reshape(torch.Tensor(Y[:n_training]),(n_training,1))

trainset = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

#%%
in_shape = (X.shape[1],X.shape[2])

mean_brain = np.mean(X,0)
mean_brain = gaussian_filter(X[2,:,:], sigma=4)
print(mean_brain.shape)

reg_net.train(False)

init_model = None
if init_model is None:
  atlas_building_net = atlas_building_model(reg_net = reg_net, init_atlas= mean_brain)
else:
  atlas_building_net = atlas_building_model.load_from_checkpoint(init_model)  

n_epochs = 50

atlas_building_trainer = pl.Trainer(gpus=1, 
                           max_epochs=n_epochs,
                           callbacks=[RichProgressBar()])

atlas_building_trainer.fit(atlas_building_net, trainloader) 

atlas_building_trainer.save_checkpoint(home+'/Sync-Exp/atlas_model_'+str(n_epochs)+'_nf'+str(n_features)+'.ckpt') 

#%%


#%%

class atlas_building_model(pl.LightningModule):
  def __init__(self, shape, int_steps = 7, init_atlas = None, n_features = 8):
    super().__init__()  
    self.shape = shape

    self.unet_model = Unet(n_features = n_features)
    self.transformer = SpatialTransformer(size=shape)
    if init_atlas is None:
      self.atlas = nn.Parameter(torch.unsqueeze(torch.randn(shape) / 10,0))
    else:
      self.atlas = nn.Parameter(torch.unsqueeze(torch.Tensor(init_atlas),0))
    self.int_steps = int_steps
    self.vecint = VecInt(inshape=shape, nsteps=int_steps)
    self.e = 0

    self.lambda_similarity = 1
    self.lambda_grad_flow  = 1
    self.lambda_magn_flow  = 0.01
    self.lambda_mean_flow  = 1
    

  def forward(self,source): # takes only one image, i.e. batch size = 1
    target = torch.unsqueeze(self.atlas,0)
    x = torch.cat([source,target],dim=1) 
    forward_velocity = self.unet_model(x)
    
    backward_velocity = -forward_velocity
    if self.int_steps > 0:
      forward_flow = self.vecint(forward_velocity)
      backward_flow= self.vecint(backward_velocity)
    
    y_source = self.transformer(source, forward_flow)
    y_target = self.transformer(target, backward_flow)
    
    return y_source, y_target, forward_flow 

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    source_batch, _ = batch

    loss = 0
    target = torch.unsqueeze(self.atlas,0)
    sum_flows = torch.zeros((1,2*source_batch.shape[2]*source_batch.shape[3]),device=self.device)
    n_images = torch.tensor(source_batch.shape[0],device=self.device)
    
    for i in range(n_images):
      source = torch.unsqueeze(source_batch[i],0)
      
      y_source,y_target,forward_flow = self(source)
      
      loss += self.lambda_similarity * F.mse_loss(target,y_source)# + F.mse_loss(y_target,source)  #Dey does the opposite (warp the atlas onto the images)
      loss += self.lambda_grad_flow * Grad().loss(forward_flow) 
      loss += self.lambda_magn_flow * F.mse_loss(torch.zeros(forward_flow.shape,device=self.device),forward_flow)
        
      sum_flows += torch.reshape(forward_flow,(1,-1)) 
      
    loss += loss / n_images + self.lambda_mean_flow * F.mse_loss(torch.zeros(sum_flows.shape,device=self.device), sum_flows / n_images) #torch.norm(sum_flows / n_images,p=2)

    return loss       
  
  def training_epoch_end(self, training_step_outputs):
    self.e += 1
    #all_preds = torch.stack(training_step_outputs)
    if (self.e-1) % 10 == 0: 
      min_val = torch.min(self.atlas[0,:,:])
      max_val = torch.max(self.atlas[0,:,:])      
      #save_image((self.atlas[0,:,:]-min_val)*255/(max_val-min_val), home+'/Sync-Exp/atlas_'+str(self.e).zfill(4)+'.png')
      imsave(home+'/Sync-Exp/atlas_'+str(self.e).zfill(4)+'.png',((self.atlas[0,:,:]-min_val)*255/(max_val-min_val)).cpu().detach().numpy().astype(np.uint8))
      
#%%
# Prepare data for pytorch
batch_size = 128

n_training = X.shape[0]
x_train = torch.reshape(torch.Tensor(X[:n_training, ...]),(n_training,1,X.shape[1],X.shape[2]))
y_train = torch.reshape(torch.Tensor(Y[:n_training]),(n_training,1))

trainset = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

#%%
in_shape = (X.shape[1],X.shape[2])

mean_brain = np.mean(X,0)
mean_brain = gaussian_filter(X[2,:,:], sigma=4)
print(mean_brain.shape)

n_features = 16

init_model = None
if init_model is None:
  atlas_building_net = atlas_building_model(shape=in_shape, n_features = n_features, init_atlas= mean_brain)
else:
  atlas_building_net = atlas_building_model.load_from_checkpoint(init_model, shape=in_shape, n_features = n_features)  
#%%
template = atlas_building_net.atlas
print(template.shape)
if display_plot:
  plt.figure()
  plt.imshow(template[0,...].cpu().detach().numpy(),cmap='gray')
  plt.show(block=False)
#%%

n_epochs = 2000

atlas_building_trainer = pl.Trainer(gpus=1, 
                           max_epochs=n_epochs,
                           callbacks=[RichProgressBar()])

atlas_building_trainer.fit(atlas_building_net, trainloader) 

atlas_building_trainer.save_checkpoint(home+'/Sync-Exp/atlas_model_'+str(n_epochs)+'_nf'+str(n_features)+'.ckpt') 

template = atlas_building_net.atlas
print(template.shape)
if display_plot:
  plt.figure()
  plt.imshow(template[0,...].cpu().detach().numpy(),cmap="gray")
  plt.show(block=False)

  plt.figure()
  plt.imshow(mean_brain - template[0,...].cpu().detach().numpy(),cmap="gray")
  plt.show(block=False)
#%%
if display_plot:
  plt.figure()
  plt.imshow(mean_brain,cmap="gray")
  plt.show(block=False)