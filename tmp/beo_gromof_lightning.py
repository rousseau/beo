#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
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

dataset = 'HCP' #HCP or IXI or Bazin  
patch_size = 64 

if dataset == 'IXI':  
  X_train = joblib.load(home+'/Exp/IXI_T1_2D_'+str(patch_size)+'_training.pkl')
  X_train = np.moveaxis(X_train,3,1)  
  Y_train = joblib.load(home+'/Exp/IXI_T2_2D_'+str(patch_size)+'_training.pkl')
  Y_train = np.moveaxis(Y_train,3,1)  
  
if dataset == 'HCP':  
  X_train = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_training.pkl')
  X_train = np.moveaxis(X_train,3,1)  
  Y_train = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_training.pkl')
  Y_train = np.moveaxis(Y_train,3,1)  
  X_test = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_testing.pkl')
  X_test = np.moveaxis(X_test,3,1)  
  Y_test = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_testing.pkl')
  Y_test = np.moveaxis(Y_test,3,1)  

if dataset == 'Bazin':
  X_train = joblib.load(home+'/Exp/BlockFace_'+str(patch_size)+'_training.pkl')
  X_train = np.moveaxis(X_train,3,1)  
  Y_train = joblib.load(home+'/Exp/Thio_'+str(patch_size)+'_training.pkl')
  Y_train = np.moveaxis(Y_train,3,1)  

n_training_samples = X_train.shape[0]
batch_size = 32 
trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=32)

testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=32)

#%%
print('setting parameters')

output_path = home+'/Sync/Experiments/'+dataset+'/'
prefix = 'gromof'

lambda_x2y = 1
lambda_y2x = 1  
lambda_cycle = 0
lambda_id = 1 
lambda_ae = 1   
lambda_v  = 1 
lambda_Dv = 1

lw = {}
lw['rx2y']= lambda_x2y #loss weight direct mapping
lw['ry2x'] = lambda_y2x #loss weight inverse mapping
lw['rx2x'] = lambda_id #loss weight identity mapping on x 
lw['ry2y'] = lambda_id #loss weight identity mapping on y
lw['rx2y2x'] = lambda_cycle #loss weight cycle for x
lw['ry2x2y'] = lambda_cycle #loss weight cycle for y
lw['idx2x'] = lambda_ae #loss weight autoencoder for x
lw['idy2y'] = lambda_ae #loss weight autoencoder for y
lw['normv'] = lambda_v  #loss weight for norm v
lw['normdv'] = lambda_Dv  #loss weight for norm Dv

n_filters = 16
n_channelsX = 1 
n_channelsY = 1 

n_layers_list = [2,4]#,2,4,8,16]#[1,2,4,8,16,32,64]
n_layers = np.max(n_layers_list)           #number of layers for the numerical integration scheme (ResNet)

reversible_mapping = 0 
shared_blocks = 0
backward_order = 1   #Behrmann et al. ICML 2019 used 100
scaling_weights = 1 

mse = torch.nn.functional.mse_loss

#%%
from pytorch_lightning.callbacks import Callback

training_loss = {}
training_loss['loss'] = []
for k in lw.keys():
  training_loss[k] = []
#current_epoch = 0 

class MyCallback(Callback):
  def on_epoch_start(self, trainer, pl_module):
    for k in training_loss.keys():
      training_loss[k].append(0.0)

  def on_epoch_end(self, trainer, pl_module):
    #current_epoch += 1
    for k in training_loss.keys():
      training_loss[k][-1] = training_loss[k][-1] / (1.0*n_training_samples)
      #print('-> epoch '+str(current_epoch)+', training '+str(k)+' : '+str(training_loss[k][-1]))
      print('-> training '+str(k)+' : '+str(training_loss[k][-1]))

#%%
from beo_lipschitz import block_lipschitz

def norm_v(feature_net,forward_blocks, x):
  n = len(forward_blocks)
  norm_velocity = torch.zeros([n])

  #var_x = torch.autograd.Variable(x, requires_grad=True)  
  #fx = feature_net(var_x)
  fx = feature_net(x)
  
  for l in range(n):  
    mx = forward_blocks[l](fx)
    vx = mx - fx
    #Max of the infinity norm of vx, which is the incremental update at each forward block
    norm_velocity[l] = torch.max( torch.norm( torch.reshape(vx,(mx.shape[0],-1)), p=np.inf, dim=1 ) )    
    fx = mx
    
  return norm_velocity


#%%
print('setting networks')

from beo_nets import feature_model, recon_model, block_mapping_model, forward_block_model
from beo_nets import backward_block_model, mapping_model


class generator_model(pl.LightningModule):
  def __init__(self, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model):
    #super(generator_model, self).__init__()
    super().__init__()   
    self.feature_model = feature_model
    self.mapping_x_to_y = mapping_x_to_y
    self.mapping_y_to_x = mapping_y_to_x
    self.reconstruction_model = reconstruction_model

  def forward(self,x,y):
    fx = self.feature_model(x)
    fy = self.feature_model(y)

    #Forward     
    mx2y = self.mapping_x_to_y(fx)
    rx2y = self.reconstruction_model(mx2y)

    #Backward
    my2x = self.mapping_y_to_x(fy)
    ry2x = self.reconstruction_model(my2x)

    #Identity mapping constraints
    my2y = self.mapping_x_to_y(fy)
    ry2y = self.reconstruction_model(my2y)

    mx2x = self.mapping_y_to_x(fx)
    rx2x = self.reconstruction_model(mx2x) 

    #Cycle consistency
    frx2y = self.feature_model(rx2y)
    mx2y2x = self.mapping_y_to_x(frx2y)
    rx2y2x = self.reconstruction_model(mx2y2x)  

    fry2x = self.feature_model(ry2x)
    my2x2y = self.mapping_x_to_y(fry2x)
    ry2x2y = self.reconstruction_model(my2x2y)  

    #Autoencoder constraints
    idy2y = self.reconstruction_model(fy)
    idx2x = self.reconstruction_model(fx)

    return rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
    return optimizer

  def training_step(self, batch, batch_idx):
    x, y = batch
    rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = self(x,y)

    loss = {}
    for k in lw.keys(): #compute only loss whose weight is non zero
      if lw[k] > 0:
        if k == 'normv':
          loss[k] = torch.sum( norm_v(feature_net,forward_blocks, x) )
          #loss[k] = loss_sobolev_norm_v(feature_net,forward_blocks, block_x2y_list, n_filters, x)              
        elif k == 'normdv':
          loss[k] = torch.sum( block_lipschitz(block_x2y_list,n_filters) )
        else:
          loss[k] = mse(eval(k),eval(k[-1]))
      else:
        loss[k] = torch.Tensor([0])[0]

    total_loss = 0
    for k in loss.keys():
      if lw[k] > 0:
        total_loss += lw[k] * loss[k]

    training_loss['loss'][-1] += total_loss.item() * x.shape[0]

    for k in loss.keys():
      training_loss[k][-1] += loss[k].item() * x.shape[0]


    #loss = F.mse_loss(rx2y,y) + F.mse_loss(ry2x,x) + F.mse_loss(rx2x,x) + F.mse_loss(ry2y,y)
    #loss+= F.mse_loss(rx2y2x,x) + F.mse_loss(ry2x2y,y) + F.mse_loss(idx2x,x) + F.mse_loss(idy2y,y)
    return total_loss

#%%

forward_blocks = torch.nn.ModuleList()#[]
backward_blocks= torch.nn.ModuleList()#[]
block_x2y_list = torch.nn.ModuleList()#[]

if shared_blocks == 1:
  if reversible_mapping == 1:
    block_f_x2y = block_mapping_model((int)(n_filters/2))
    block_g_x2y = block_mapping_model((int)(n_filters/2))
    block_x2y = torch.nn.ModuleList()
    block_x2y.append(block_f_x2y)
    block_x2y.append(block_g_x2y)
    #block_x2y = [block_f_x2y,block_g_x2y]
    
  else:
    block_x2y = torch.nn.ModuleList()
    block_x2y.append(block_mapping_model(n_filters))
    #block_x2y = [block_mapping_model(n_filters)]
    
  forward_block = forward_block_model(block=block_x2y)
  backward_block = backward_block_model(block=block_x2y, order = backward_order)    
    
  for l in range(n_layers):
    forward_blocks.append(forward_block)
    backward_blocks.append(backward_block)
    block_x2y_list.append(block_x2y)


else:
  for l in range(n_layers):
    if reversible_mapping == 1:
      block_f_x2y = block_mapping_model((int)(n_filters/2))
      block_g_x2y = block_mapping_model((int)(n_filters/2))
      block_x2y = torch.nn.ModuleList()
      block_x2y.append(block_f_x2y)
      block_x2y.append(block_g_x2y)
      #block_x2y = [block_f_x2y,block_g_x2y]
      
    else:
      block_x2y = torch.nn.ModuleList()
      block_x2y.append(block_mapping_model(n_filters))
      #block_x2y = [block_mapping_model(n_filters)]
      
    forward_block = forward_block_model(block=block_x2y)
    backward_block = backward_block_model(block=block_x2y, order = backward_order)    
    
    forward_blocks.append(forward_block)
    backward_blocks.append(backward_block)
    block_x2y_list.append(block_x2y)

  
feature_net = feature_model(n_channelsX, n_filters)
recon_net = recon_model(n_filters, n_channelsY)
    

#%%
print('training ...')
iteration = 0


for nl in range(len(n_layers_list)):
  print('*********************** NEW SCALE **********************************')
  n_blocks = n_layers_list[nl]
  print(str(len(forward_blocks[0:n_blocks]))+' mapping layers ')

  if scaling_weights == 1:
    if iteration > 0:
      scaling = n_layers_list[nl-1] * 1.0 / n_layers_list[nl]
      print('Doing weight scaling by '+str( scaling ))
      for blocks in block_x2y_list:
        for block in blocks:  
          for layername in block.state_dict().keys():
            if 'conv' in layername:
              weights = block.state_dict()[layername] #torch tensor
              block.state_dict()[layername].copy_(weights*scaling)

  
  mx2y_net = mapping_model(forward_blocks[0:n_blocks])
  my2x_net = mapping_model(backward_blocks[0:n_blocks])

  net = generator_model(feature_net, mx2y_net, my2x_net, recon_net)

  print('feature net    -- number of trainable parameters : '+str( sum(p.numel() for p in feature_net.parameters() if p.requires_grad) ))
  print('mapping x to y -- number of trainable parameters : '+str( sum(p.numel() for p in mx2y_net.parameters() if p.requires_grad) ))
  print('mapping y to x -- number of trainable parameters : '+str( sum(p.numel() for p in my2x_net.parameters() if p.requires_grad) ))
  print('recon net      -- number of trainable parameters : '+str( sum(p.numel() for p in recon_net.parameters() if p.requires_grad) ))
  print('net            -- number of trainable parameters : '+str( sum(p.numel() for p in net.parameters() if p.requires_grad) ))

  trainer = pl.Trainer(gpus=1, max_epochs=2, callbacks=[MyCallback()])
  trainer.fit(net, trainloader)
  iteration += 1 

#%%
#Save hist figure
plt.figure(figsize=(4,4))

for k in training_loss.keys():
  plt.plot(training_loss[k])

plt.ylabel('training loss')  
plt.xlabel('epochs')
plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$','$\|\| Dv \|\|_{1,\infty}$'], loc='upper right')
plt.savefig(output_path+prefix+'_loss_training.png',dpi=150, bbox_inches='tight')
plt.close()

#%%
print('visualization')
#Take 10 examples for visualization
n = 10
n_testing_samples = X_test.shape[0]
step = (int)(n_testing_samples / n)
t1_vis = torch.Tensor(X_test[0:n*step:step,:,:,:])
t2_vis = torch.Tensor(Y_test[0:n*step:step,:,:,:])

net.eval()
[a,b,c,d,e,f,g,h] = net(t1_vis,t2_vis)

from beo_visu import show_patches

show_patches(patch_list=[t1_vis.cpu().detach().numpy(),
                        t2_vis.cpu().detach().numpy(),
                        a.cpu().detach().numpy(),
                        b.cpu().detach().numpy(),
                        c.cpu().detach().numpy(),
                        d.cpu().detach().numpy(),
                        e.cpu().detach().numpy(),
                        f.cpu().detach().numpy(),
                        g.cpu().detach().numpy(),
                        h.cpu().detach().numpy()],
            titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'],
            filename=output_path+prefix+'_fig_patches.png')
