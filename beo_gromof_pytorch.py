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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

n_testing_samples = X_test.shape[0]
#Take 10 examples for visualization
n = 10
step = (int)(n_testing_samples / n)
t1_vis = torch.Tensor(X_test[0:n*step:step,:,:,:]).to(device)
t2_vis = torch.Tensor(Y_test[0:n*step:step,:,:,:]).to(device)

#%%
n_filters = 32
n_channelsX = 1
n_channelsY = 1 

n_epochs = 5       # epochs per multiscale step
n_layers_list = [8]#,2,4,8,16]#[1,2,4,8,16,32,64]
n_layers = np.max(n_layers_list)           #number of layers for the numerical integration scheme (ResNet)
batch_size = 32

shared_blocks = 0
reversible_mapping = 0
backward_order = 1
scaling_weights = 1

output_path = home+'/Sync/Experiments/'+dataset+'/'
prefix = 'gromof'

lambda_x2y = 1
lambda_y2x = 0
lambda_cycle = 0
lambda_id = 0
lambda_ae = 1
lambda_v = 0.1

prefix += '_epochs_'+str(n_epochs)
prefix += '_nl'
for l in n_layers_list:
  prefix += '_'+str(l)
prefix += '_nf_'+str(n_filters)
prefix += '_loss_'+str(lambda_x2y)+'_'+str(lambda_y2x)+'_'+str(lambda_cycle)+'_'+str(lambda_id)+'_'+str(lambda_ae)+'_'+str(lambda_v)
prefix += '_shared_'+str(shared_blocks)
prefix += '_rev_'+str(reversible_mapping)
prefix += '_bo_'+str(backward_order)

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

#mx2y_net = mapping_model(forward_blocks).to(device)
#my2x_net = mapping_model(forward_blocks).to(device)
#net = generator_model(feature_net, mx2y_net, my2x_net, recon_net).to(device)

#%%

#Using hook to get activation at different steps of the mapping x to y
# activation = {}
# def get_activation(name):
#   def hook(model, input, output):
#     activation[name] = output.detach()
#   return hook

# for l in range(n_layers):
#   net.mapping_x_to_y.blocks[l].register_forward_hook(get_activation('block'+str(l)))



#%%
trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


  
#%%
#from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.ylim(bottom = 0, top=1) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    print(layers)
    
#%%
class intermediate_mapping_model(torch.nn.Module):
  def __init__(self, feature_model, mapping_model):
    super(intermediate_mapping_model, self).__init__()   
    self.feature_model = feature_model
    self.mapping_model = mapping_model

  def forward(self,x):
    fx = self.feature_model(x)
    mx2y = self.mapping_model(fx)

    return mx2y

#%%


# def norm_v():
#   norm_velocity = 0
#   for l in range(n_layers):
#     act = activation['block'+str(l)].cpu().numpy()
#     current_max = np.max(np.linalg.norm(act.reshape((act.shape[0],-1)), ord=np.inf,axis=1))
#     norm_velocity = np.maximum(current_max, norm_velocity)
#   return norm_velocity  

# def norm_Dv(x):
#   norm_gradient_velocity = 0
#   var_x = torch.autograd.Variable(x, requires_grad=True)  
#   fx_tmp = feature_net(var_x)
#   for l in range(n_layers):  
#     m_tmp = mapping_model(forward_blocks[:(l+1)]).to(device)
#     mx = m_tmp(fx_tmp)
  
#     gradients = torch.autograd.grad(outputs=mx, inputs=var_x,
#                                     grad_outputs=torch.ones(mx.size()).to(device),
#                           create_graph=True, retain_graph=True, only_inputs=True)[0].cpu().detach().numpy()
    
#     current_max = np.max(np.linalg.norm(gradients.reshape((gradients.shape[0],-1)), ord=np.inf,axis=1))
#     norm_gradient_velocity = np.maximum(current_max, norm_gradient_velocity)
#   return norm_gradient_velocity

def norm_inf_v(feature_model,forward_blocks, x, along_trajectory = True, only_activation = False):
  norm_velocity = torch.Tensor([0]).to(device)
  norm_gradient_velocity = torch.Tensor([0]).to(device)

  var_x = torch.autograd.Variable(x, requires_grad=True)  
  fx = feature_model(var_x)
  
  for l in range(len(forward_blocks)):  
    mx = forward_blocks[l](fx)
   
    current_max = torch.max( torch.norm( mx.view(mx.shape[0],-1), p=np.inf, dim=1 ) )
    norm_velocity = torch.max(current_max, norm_velocity)
    
    if only_activation is False:
      gradients = torch.autograd.grad(outputs=mx, inputs=fx,
                                      grad_outputs=torch.ones(mx.size()).to(device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]#.cpu().detach().numpy()
      
      current_max = torch.max( torch.norm( gradients.view(gradients.shape[0],-1), p=np.inf, dim=1 ) )
      norm_gradient_velocity = torch.max(current_max, norm_gradient_velocity)
    
    #if sampling along trajectory otherwise compute only over fx
    if along_trajectory is True:
      fx = mx
    
  return norm_velocity + norm_gradient_velocity

#%%
# #Just x to y
# lambda_direct = 0
# lambda_v = 1
# batch_size = 128

# class generator_x2y_model(torch.nn.Module):
#   def __init__(self, feature_model, mapping_x_to_y, reconstruction_model):
#     super(generator_x2y_model, self).__init__()   
#     self.feature_model = feature_model
#     self.mapping_x_to_y = mapping_x_to_y
#     self.reconstruction_model = reconstruction_model

#   def forward(self,x,y):
#     fx = self.feature_model(x)
#     mx2y = self.mapping_x_to_y(fx)
#     rx2y = self.reconstruction_model(mx2y)

#     return rx2y

# net_x2y = generator_x2y_model(feature_net, mx2y_net, recon_net).to(device)
# net_x2y = torch.nn.DataParallel(net_x2y)

# for epoch in range(n_epochs):
#   training_loss = 0.0
#   training_loss1 = 0    
#   training_loss_nv = 0

#   with tqdm(total = n_training_samples, desc=f'Epoch {epoch + 1}', file=sys.stdout) as pbar:

#     net.train()
#     for (x,y) in trainloader:
#       x, y = x.to(device), y.to(device)

#       optimizer.zero_grad()

#       rx2y = net_x2y(x,y)
#       loss1 = mse(rx2y, y)
      
#       loss_nv = norm_inf_v(x)
#       loss = lambda_direct * loss1 + lambda_v * loss_nv
      
#       loss.backward()
 
#       optimizer.step()

#       training_loss += loss.item() * x.shape[0]
#       training_loss1 += loss1.item() * x.shape[0]

#       training_loss_nv += loss_nv.item() * x.shape[0]   
      

#       pbar.update(x.shape[0])

#   print('\n -> epoch '+str(epoch + 1)+', training loss : '+str(training_loss / n_training_samples))

#   print(' -> epoch '+str(epoch + 1)+', training loss 1: '+str(training_loss1 / n_training_samples))

#   print(' -> epoch '+str(epoch + 1)+', training loss nv: '+str(training_loss_nv / n_training_samples))


#%%
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

max_epochs = n_epochs * len(n_layers_list)
norm_velocity = np.zeros( (n_layers , max_epochs) )
norm_gradient_velocity = np.zeros( (n_layers, max_epochs) )  


iteration = 0

training_loss = {}
training_loss['loss'] = []
for k in lw.keys():
  training_loss[k] = []

validation_loss = {}
validation_loss['loss'] = []
for k in lw.keys():
  validation_loss[k] = []
validation_loss['reversibility'] = []


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

  #Build nets
  intermediate_models = []
  for l in range(n_layers):
    intermediate_models.append( intermediate_mapping_model(feature_net,mapping_model(forward_blocks[:(l+1)])).to(device) )

  mx2y_net = mapping_model(forward_blocks[0:n_blocks]).to(device)
  my2x_net = mapping_model(forward_blocks[0:n_blocks]).to(device)

  net = generator_model(feature_net, mx2y_net, my2x_net, recon_net).to(device)

#  if torch.cuda.device_count() > 1:
    #print("Let's use", torch.cuda.device_count(), "GPUs!")
    #net = torch.nn.DataParallel(net)
    #feature_net = torch.nn.DataParallel(feature_net)
    #mx2y_net = torch.nn.DataParallel(mx2y_net)
    # my2x_net = torch.nn.DataParallel(my2x_net)
    # recon_net = torch.nn.DataParallel(recon_net)
    # batch_size *= torch.cuda.device_count()

  print('feature net    -- number of trainable parameters : '+str( sum(p.numel() for p in feature_net.parameters() if p.requires_grad) ))
  print('mapping x to y -- number of trainable parameters : '+str( sum(p.numel() for p in mx2y_net.parameters() if p.requires_grad) ))
  print('mapping y to x -- number of trainable parameters : '+str( sum(p.numel() for p in my2x_net.parameters() if p.requires_grad) ))
  print('recon net      -- number of trainable parameters : '+str( sum(p.numel() for p in recon_net.parameters() if p.requires_grad) ))
  print('net            -- number of trainable parameters : '+str( sum(p.numel() for p in net.parameters() if p.requires_grad) ))

  mse = torch.nn.functional.mse_loss
  optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

  for epoch in range(n_epochs):

    print('\n Training ')

    for k in training_loss.keys():
      training_loss[k].append(0.0)

    with tqdm(total = n_training_samples, desc=f'Epoch {epoch + 1}', file=sys.stdout) as pbar: 
      net.train()
      for (x,y) in trainloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = net(x,y)
        loss = {}
        for k in lw.keys(): #compute only loss whose weight is non zero
          if lw[k] > 0:
            if k == 'normv':
              loss[k] = norm_inf_v(feature_net,forward_blocks,x)[0]
            else:
              loss[k] = mse(eval(k),eval(k[-1]))
          else:
            loss[k] = torch.Tensor([0]).to(device)[0]
        # loss['rx2y'] = mse(rx2y, y)
        # loss['ry2x'] = mse(ry2x, x)
        # loss['rx2x'] = mse(rx2x, x)
        # loss['ry2y'] = mse(ry2y, y)
        # loss['rx2y2x'] = mse(rx2y2x, x)
        # loss['ry2x2y'] = mse(ry2x2y, y)
        # loss['idx2x'] = mse(idx2x, x)
        # loss['idy2y'] = mse(idy2y, y)    
        # if lw['normv'] > 0:
        #   loss['normv'] = norm_inf_v(intermediate_models,x)[0]
        # else:
        #   loss['normv'] = torch.Tensor([0]).to(device)[0]
        total_loss = 0
        for k in loss.keys():
          total_loss += lw[k] * loss[k]
              
        total_loss.backward()

        optimizer.step()

        training_loss['loss'][-1] += total_loss.item() * x.shape[0]

        for k in loss.keys():
          training_loss[k][-1] += loss[k].item() * x.shape[0]
        
        pbar.update(x.shape[0])

    for k in training_loss.keys():
      training_loss[k][-1] = training_loss[k][-1] / (1.0*n_training_samples)
      print('-> epoch '+str(epoch + 1)+', training '+str(k)+' : '+str(training_loss[k][-1]))

    #Save hist figure
    plt.figure(figsize=(4,4))

    for k in training_loss.keys():
      plt.plot(training_loss[k])

    plt.ylabel('training loss')  
    plt.xlabel('epochs')
    plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$'], loc='upper right')
    plt.savefig(output_path+prefix+'_loss_training.png',dpi=150, bbox_inches='tight')
    plt.close()

    print('\n Validation ')
    net.eval()  

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
                 filename=output_path+prefix+'_current'+str(iteration)+'fig_patches.png')

    [a,b,c,d,e,f,g,h] = net(t1_vis,t2_vis)  
    show_patches(patch_list=[t1_vis.cpu().detach().numpy(),
                             t2_vis.cpu().detach().numpy(),
                             a.cpu().detach().numpy() - t2_vis.cpu().detach().numpy(),
                             b.cpu().detach().numpy() - t1_vis.cpu().detach().numpy(),
                             c.cpu().detach().numpy() - t1_vis.cpu().detach().numpy(),
                             d.cpu().detach().numpy() - t2_vis.cpu().detach().numpy(),
                             e.cpu().detach().numpy() - t1_vis.cpu().detach().numpy(),
                             f.cpu().detach().numpy() - t2_vis.cpu().detach().numpy(),
                             g.cpu().detach().numpy() - t1_vis.cpu().detach().numpy(),
                             h.cpu().detach().numpy() - t2_vis.cpu().detach().numpy()],
                 titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'],
                 filename=output_path+prefix+'_current'+str(iteration)+'fig_error_patches.png')


    for k in validation_loss.keys():
      validation_loss[k].append(0.0)
      
    for (x,y) in testloader:
      x, y = x.to(device), y.to(device)
      fx_tmp = feature_net(x)
      mx_tmp = mx2y_net(fx_tmp)
      my_tmp = my2x_net(mx_tmp)
      validation_loss['reversibility'][-1] += mse(fx_tmp,my_tmp).item() * x.shape[0]
      
      rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = net(x,y)
      loss = {}
      for k in lw.keys(): #compute only loss whose weight is non zero
        if k == 'normv':
          loss[k] = norm_inf_v(feature_net,forward_blocks,x)[0]
        else:
          loss[k] = mse(eval(k),eval(k[-1]))

      total_loss = 0
      for k in loss.keys():
        total_loss += lw[k] * loss[k]
            
      validation_loss['loss'][-1] += total_loss.item() * x.shape[0]

      for k in loss.keys():
        validation_loss[k][-1] += loss[k].item() * x.shape[0]
      

    for k in validation_loss.keys():
      validation_loss[k][-1] /= n_testing_samples
      print('-> epoch '+str(epoch + 1)+', validation '+str(k)+' : '+str(validation_loss[k][-1]))

    #Save hist figure
    plt.figure(figsize=(4,4))
 
    for k in validation_loss.keys():
      if k != 'reversibility':
        plt.plot(validation_loss[k])

    plt.ylabel('validation loss')  
    plt.xlabel('epochs')
    plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$'], loc='upper right')
    plt.savefig(output_path+prefix+'_loss_validation.png',dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure()
    plt.plot(validation_loss['reversibility'])
    plt.legend(['$\| f(x) - m^{-1} \circ m \circ f(x) \|^2$'])
    plt.savefig(output_path+prefix+'_inverr.png',dpi=150, bbox_inches='tight')
    plt.close()   

    print('\nComputing norm velocity...')  
    for (x,y) in testloader:
      x, y = x.to(device), y.to(device)
      
      var_x = torch.autograd.Variable(x, requires_grad=True)  
      
      for l in range(len(intermediate_models)):  
        mx = intermediate_models[l](var_x)
       
        current_max = torch.max( torch.norm( mx.view(mx.shape[0],-1), p=np.inf, dim=1 ) ).cpu().detach().numpy()
        norm_velocity[l,iteration] = np.maximum(current_max, norm_velocity[l,iteration])
              
        gradients = torch.autograd.grad(outputs=mx, inputs=var_x,
                                        grad_outputs=torch.ones(mx.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]#.cpu().detach().numpy()
        
        current_max = torch.max( torch.norm( gradients.view(gradients.shape[0],-1), p=np.inf, dim=1 ) ).cpu().detach().numpy()
        norm_gradient_velocity[l,iteration] = np.maximum(current_max, norm_gradient_velocity[l,iteration])
      

    plt.figure(figsize=(4,4))
    
    for i in range(norm_velocity.shape[0]):
      plt.plot(norm_velocity[i,0:iteration+1])
    plt.savefig(output_path+prefix+'_norm_velocity.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))    
    im = plt.imshow(norm_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    for i in range(norm_gradient_velocity.shape[0]):
      plt.plot(norm_gradient_velocity[i,0:iteration+1])
    plt.savefig(output_path+prefix+'_norm_gradient_velocity.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    im = plt.imshow(norm_gradient_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_gradient_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

    
    #Calcul de Lipschitz
    iteration += 1 
    
print('last iteration : '+str(iteration))

#%%

sys.exit()

#%%
norm_velocity = np.zeros((n_layers, n_epochs))

norm_gradient_velocity = np.zeros((n_layers, n_epochs))


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
  
  training_loss_nv = 0
  
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
      loss_nv = norm_inf_v(x)
      loss = lw1 * loss1 + lw2 * loss2 + lw3 * loss3 + lw4 * loss4 + lw5 * loss5 + lw6 * loss6 + lw7 * loss7 + lw8 * loss8 + lambda_v * loss_nv
            
      loss.backward()
 
      #to check vanishing or exploding gradient
      #plot_grad_flow(net.named_parameters())

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

      training_loss_nv += loss_nv.item() * x.shape[0]   
      

      pbar.update(x.shape[0])
      

    net.eval()  
    # print('\nComputing validation loss...')  
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

    print('\nComputing norm velocity...')  
    for (x,y) in testloader:
      x, y = x.to(device), y.to(device)
      rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = net(x,y)
 
  
 
      for l in range(n_layers):
        act = activation['block'+str(l)].cpu().numpy()
        current_max = np.max(np.linalg.norm(act.reshape((act.shape[0],-1)), ord=np.inf,axis=1))
        norm_velocity[l,epoch] = np.maximum(current_max, norm_velocity[l,epoch])
      
        var_x = torch.autograd.Variable(x, requires_grad=True)  
        fx_tmp = feature_net(var_x)
        m_tmp = mapping_model(forward_blocks[:(l+1)]).to(device)
        mx = m_tmp(fx_tmp)

        #Provide same result as using hook -> good !
        #act = mx.cpu().detach().numpy()
        #current_max = np.max(np.linalg.norm(act.reshape((act.shape[0],-1)), ord=np.inf,axis=1))
        #norm_velocity2[l,epoch] = np.maximum(current_max, norm_velocity2[l,epoch])

        gradients = torch.autograd.grad(outputs=mx, inputs=var_x,
                                        grad_outputs=torch.ones(mx.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0].cpu().detach().numpy()

        
        # var_act = torch.autograd.Variable(activation['block'+str(l)], requires_grad=True)
        # gradients = torch.autograd.grad(outputs=var_act, inputs=var_x,
        #                                 grad_outputs=torch.ones(var_act.size()).cuda(0) if use_cuda else torch.ones(
        #                           var_act.size()),
        #                       create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0].cpu().numpy()
        
        
        current_max = np.max(np.linalg.norm(gradients.reshape((gradients.shape[0],-1)), ord=np.inf,axis=1))
        norm_gradient_velocity[l,epoch] = np.maximum(current_max, norm_gradient_velocity[l,epoch])



    plt.figure(figsize=(4,4))
    
    for i in range(norm_velocity.shape[0]):
      plt.plot(norm_velocity[i,0:epoch+1])
    plt.savefig(output_path+prefix+'_norm_velocity.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))    
    im = plt.imshow(norm_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    for i in range(norm_gradient_velocity.shape[0]):
      plt.plot(norm_gradient_velocity[i,0:epoch+1])
    plt.savefig(output_path+prefix+'_norm_gradient_velocity.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    im = plt.imshow(norm_gradient_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_gradient_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

      
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

  print(' -> epoch '+str(epoch + 1)+', training loss nv: '+str(training_loss_nv / n_training_samples))

  print(' -> epoch '+str(epoch + 1)+', reversibility loss : '+str(reversibility_loss / n_training_samples))


  print(' -> epoch '+str(epoch + 1)+', validation loss : '+str(valid_loss / n_training_samples))

print('Training Done')    

# %%

#torch.save(net,output_path+prefix+'_net')