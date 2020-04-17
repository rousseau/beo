#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
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


#%%
import torch

class conv_relu_block_2d(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(conv_relu_block_2d, self).__init__()
    self.relu = torch.nn.ReLU()
    self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    return x

class feature_model(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(feature_model, self).__init__()
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()
    self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
    self.convblock = conv_relu_block_2d(out_channels,out_channels)
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.convblock(x)
    x = self.tanh(x)
    return x  

class recon_model(torch.nn.Module):
  def __init__(self, in_channels, out_channels):
    super(recon_model, self).__init__()
    self.relu = torch.nn.ReLU()
    self.up = torch.nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2, stride=2)
    #self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear")
    self.convblock = conv_relu_block_2d(in_channels,in_channels)
    self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)    
    
  def forward(self, x):
    x = self.up(x)
    x = self.relu(x)
    x = self.convblock(x)
    x = self.relu(x)
    x = self.conv1(x)
    return x  

class block_mapping_model(torch.nn.Module):
  def __init__(self, in_channels):
    super(block_mapping_model, self).__init__()
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()
    self.conv1 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False)    
    self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False)    
    self.conv3 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False)    

  def forward(self,x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.tanh(x)
    return x

class forward_block_model(torch.nn.Module):
  def __init__(self, block):
    super(forward_block_model, self).__init__()
    self.block = block

  def forward(self,x):
    if len(self.block) == 1:
      y = self.block[0](x)
      x = torch.add(x,y)
    else:
      x1, x2 = torch.chunk(x, 2, dim=1)
      xx = self.block[0](x2)
      y1 = torch.add(x1,xx)
      xx = self.block[1](y1)
      y2 = torch.add(x2,xx)

      x = torch.cat([y1, y2], dim=1)
    return x  
    
class backward_block_model(torch.nn.Module):
  def __init__(self, block,order=1):
    super(backward_block_model, self).__init__()
    self.block = block
    self.order = order

  def forward(self,x):
    if len(self.block) == 1:

      z = x
      for i in range(self.order):
        y = self.block[0](x)
        z = torch.sub(x,y)
      x = z
    else:
      y1, y2 = torch.chunk(x, 2, dim=1)
      yy = self.block[1](y1)
      x2 = torch.sub(y2,yy)
      yy = self.block[0](x2)
      x1 = torch.sub(y1,yy)
      
      x = torch.cat([x1, x2], dim=1)
    return x  

class mapping_model(torch.nn.Module):
  def __init__(self, blocks):
    super(mapping_model, self).__init__()
    self.blocks = blocks

  def forward(self,x):
    for i in range(len(self.blocks)):
      x = self.blocks[i](x)
    return x

class generator_model(torch.nn.Module):
  def __init__(self, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model):
    super(generator_model, self).__init__()   
    self.feature_model = feature_model
    self.mapping_x_to_y = mapping_x_to_y
    self.mapping_y_to_x = mapping_y_to_x
    self.reconstruction_model = reconstruction_model

  def forward(self,x,y):
    fx = self.feature_model(x)
    fy = self.feature_model(y)

    mx2y = self.mapping_x_to_y(fx)
    my2x = self.mapping_y_to_x(fy)

    rx2y = self.reconstruction_model(mx2y)
    ry2x = self.reconstruction_model(my2x)

    return rx2y,ry2x


# %%
batch_size = 16  

trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


#%%
n_filters = 32
n_channelsX = 1
n_channelsY = 1 

n_epochs = 2
n_layers = 2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

feature_net = feature_model(n_channelsX, n_filters).to(device)
recon_net = recon_model(n_filters, n_channelsY).to(device)

shared_blocks = 0
reversible_mapping = 0
backward_order = 1

forward_blocks = []
backward_blocks= []
block_x2y_list = []
if shared_blocks == 1:
  if reversible_mapping == 1:
    block_f_x2y = block_mapping_model((int)(n_filters/2)).to(device)
    block_g_x2y = block_mapping_model((int)(n_filters/2)).to(device)
    block_x2y = [block_f_x2y,block_g_x2y]
    
  else:
    block_x2y = [block_mapping_model(n_filters).to(device)]
    
  forward_block = forward_block_model(block=block_x2y)
  backward_block = backward_block_model(block=block_x2y, order = backward_order)    
  block_x2y_list.append(block_x2y)
    
  for l in range(n_layers):
    forward_blocks.append(forward_block)
    backward_blocks.append(backward_block)

else:
  for l in range(n_layers):
    if reversible_mapping == 1:
      block_f_x2y = block_mapping_model((int)(n_filters/2)).to(device)
      block_g_x2y = block_mapping_model((int)(n_filters/2)).to(device)
      block_x2y = [block_f_x2y,block_g_x2y]
      
    else:
      block_x2y = [block_mapping_model(n_filters).to(device)]
      
    forward_block = forward_block_model(block=block_x2y)
    backward_block = backward_block_model(block=block_x2y, order = backward_order)    
    
    forward_blocks.append(forward_block)
    backward_blocks.append(backward_block)
    block_x2y_list.append(block_x2y)

mx2y_net = mapping_model(forward_blocks).to(device)
my2x_net = mapping_model(forward_blocks).to(device)


net = generator_model(feature_net, mx2y_net, my2x_net, recon_net).to(device)

print('number of trainable parameters:'+str( sum(p.numel() for p in net.parameters() if p.requires_grad) ))

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

#%%
    
for epoch in range(n_epochs):
  running_loss = 0.0
  for i, (x,y) in enumerate(trainloader, 0):
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    #outputs = net(x,y)

    out1, out2 = net(x,y)
    loss1 = criterion(out1, y)
    loss2 = criterion(out2, x)
    loss = loss1 + loss2
    loss.backward()

    #loss = criterion(outputs, y)
    #loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
      print('[%d, %5d] loss: %.3f' %
            (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0

print('Finished Training')    

# %%
