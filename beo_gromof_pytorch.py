#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import joblib
from os.path import expanduser
home = expanduser("~")
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys


dataset = 'HCP' #HCP or IXI or Bazin  
patch_size = 64 

n_filters = 32
n_channelsX = 1
n_channelsY = 1 

n_epochs = 1
n_layers = 10
batch_size = 32  

shared_blocks = 1
reversible_mapping = 0
backward_order = 1

output_path = home+'/Sync/Experiments/'+dataset+'/'
prefix = 'gromof'

lambda_direct = 1
lambda_cycle = 1
lambda_id = 0.1
lambda_ae = 1

prefix += '_epochs_'+str(n_epochs)
prefix += '_nl_'+str(n_layers)
prefix += '_nf_'+str(n_filters)
prefix += '_loss_'+str(lambda_direct)+'_'+str(lambda_cycle)+'_'+str(lambda_id)+'_'+str(lambda_ae)
prefix += '_shared_'+str(shared_blocks)
prefix += '_rev_'+str(reversible_mapping)
prefix += '_bo_'+str(backward_order)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#%%

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

def show_patches(patch_list,titles, filename):

  n_rows = patch_list[0].shape[0]
  n_cols = len(patch_list)

  vmin = -2 
  vmax = 2
  plt.figure(figsize=(2. * n_cols, 2. * n_rows))
  for i in range(n_rows):
    for j in range(n_cols):
      sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
      sub.imshow(patch_list[j][i,0,:,:],
                 cmap=plt.cm.gray,
                 interpolation="nearest",
                 vmin=vmin,vmax=vmax)
      sub.axis('off')
      if i == 0:
        sub.set_title(titles[j])
  plt.axis('off')
  plt.savefig(filename,dpi=150, bbox_inches='tight')
  plt.close()

#%%

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
    #self.up = torch.nn.ConvTranspose2d(in_channels,in_channels,kernel_size=2, stride=2)
    self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear")
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
    self.block = torch.nn.ModuleList(block)

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
    self.block = torch.nn.ModuleList(block)
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
    self.blocks = torch.nn.ModuleList(blocks)

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



#%%
feature_net = feature_model(n_channelsX, n_filters).to(device)
recon_net = recon_model(n_filters, n_channelsY).to(device)

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

print('feature net    -- number of trainable parameters : '+str( sum(p.numel() for p in feature_net.parameters() if p.requires_grad) ))
print('mapping x to y -- number of trainable parameters : '+str( sum(p.numel() for p in mx2y_net.parameters() if p.requires_grad) ))
print('mapping y to x -- number of trainable parameters : '+str( sum(p.numel() for p in my2x_net.parameters() if p.requires_grad) ))
print('recon net      -- number of trainable parameters : '+str( sum(p.numel() for p in recon_net.parameters() if p.requires_grad) ))
print('net            -- number of trainable parameters : '+str( sum(p.numel() for p in net.parameters() if p.requires_grad) ))

mse = torch.nn.functional.mse_loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# %%

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = torch.nn.DataParallel(net)
  # feature_net = torch.nn.DataParallel(feature_net)
  # mx2y_net = torch.nn.DataParallel(mx2y_net)
  # my2x_net = torch.nn.DataParallel(my2x_net)
  # recon_net = torch.nn.DataParallel(recon_net)
  # batch_size *= torch.cuda.device_count()

trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


  
#%%


lw1 = lambda_direct #loss weight direct mapping
lw2 = lambda_direct #loss weight inverse mapping
lw3 = lambda_id #loss weight identity mapping on x 
lw4 = lambda_id #loss weight identity mapping on y
lw5 = lambda_cycle #loss weight cycle for x
lw6 = lambda_cycle #loss weight cycle for y
lw7 = lambda_ae #loss weight autoencoder for x
lw8 = lambda_ae #loss weight autoencoder for y

n_training_samples = X_train.shape[0]

n_testing_samples = X_test.shape[0]
#Take 10 examples for visualization
n = 10
step = (int)(n_testing_samples / n)
t1_vis = torch.Tensor(X_test[0:n*step:step,:,:,:]).to(device)
t2_vis = torch.Tensor(Y_test[0:n*step:step,:,:,:]).to(device)

for epoch in range(n_epochs):

  training_loss = 0.0
  valid_loss = 0.0
  
  training_loss1 = 0  
  training_loss2 = 0  
  training_loss3 = 0  
  training_loss4 = 0  
  training_loss5 = 0  
  training_loss6 = 0  
  training_loss7 = 0  
  training_loss8 = 0  
  
  reversibility_loss = 0
  
  with tqdm(total = n_training_samples, desc=f'Epoch {epoch + 1}', file=sys.stdout) as pbar:

    net.train()
    for (x,y) in trainloader:
      x, y = x.to(device), y.to(device)

      optimizer.zero_grad()

      rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = net(x,y)
      loss1 = mse(rx2y, y)
      loss2 = mse(ry2x, x)
      loss3 = mse(rx2x, x)
      loss4 = mse(ry2y, y)
      loss5 = mse(rx2y2x, x)
      loss6 = mse(ry2x2y, y)
      loss7 = mse(idx2x, x)
      loss8 = mse(idy2y, y)    
      loss = lw1 * loss1 + lw2 * loss2 + lw3 * loss3 + lw4 * loss4 + lw5 * loss5 + lw6 * loss6 + lw7 * loss7 + lw8 * loss8
      loss.backward()

      optimizer.step()

      training_loss += loss.item() * x.shape[0]

      training_loss1 += loss1.item() * x.shape[0]
      training_loss2 += loss2.item() * x.shape[0]
      training_loss3 += loss3.item() * x.shape[0]
      training_loss4 += loss4.item() * x.shape[0]
      training_loss5 += loss5.item() * x.shape[0]
      training_loss6 += loss6.item() * x.shape[0]
      training_loss7 += loss7.item() * x.shape[0]
      training_loss8 += loss8.item() * x.shape[0]      

      pbar.update(x.shape[0])
      
    # print('\nComputing validation loss...')  
    # net.eval()
    # for (x,y) in testloader:
    #   x, y = x.to(device), y.to(device)

    #   rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = net(x,y)
    #   loss1 = mse(rx2y, y)
    #   loss2 = mse(ry2x, x)
    #   loss3 = mse(rx2x, x)
    #   loss4 = mse(ry2y, y)
    #   loss5 = mse(rx2y2x, x)
    #   loss6 = mse(ry2x2y, y)
    #   loss7 = mse(idx2x, x)
    #   loss8 = mse(idy2y, y)    
    #   loss = lw1 * loss1 + lw2 * loss2 + lw3 * loss3 + lw4 * loss4 + lw5 * loss5 + lw6 * loss6 + lw7 * loss7 + lw8 * loss8

    #   valid_loss += loss.item() * x.shape[0]

      

      
    [a,b,c,d,e,f,g,h] = net(t1_vis,t2_vis)  
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
                 filename=output_path+prefix+'_current'+str(epoch)+'fig_patches.png')
    
    print('checking reversibility')    
    for (x,y) in testloader:
      x, y = x.to(device), y.to(device)
      fx_tmp = feature_net(x)
      mx_tmp = mx2y_net(fx_tmp)
      my_tmp = my2x_net(mx_tmp)
      reversibility_loss += mse(fx_tmp,my_tmp).item() * x.shape[0]
    
    

  print('\n -> epoch '+str(epoch + 1)+', training loss : '+str(training_loss / n_training_samples))

  print(' -> epoch '+str(epoch + 1)+', training loss 1: '+str(training_loss1 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 2: '+str(training_loss2 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 3: '+str(training_loss3 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 4: '+str(training_loss4 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 5: '+str(training_loss5 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 6: '+str(training_loss6 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 7: '+str(training_loss7 / n_training_samples))
  print(' -> epoch '+str(epoch + 1)+', training loss 8: '+str(training_loss8 / n_training_samples))

  print(' -> epoch '+str(epoch + 1)+', reversibility loss : '+str(reversibility_loss / n_training_samples))


  print(' -> epoch '+str(epoch + 1)+', validation loss : '+str(valid_loss / n_training_samples))

print('Training Done')    

# %%

#torch.save(net,output_path+prefix+'_net')