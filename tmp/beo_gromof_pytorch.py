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
print(device)

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
 
batch_size = 32 
n_epochs = 5       # epochs per multiscale step

n_layers_list = [2,4,8,16]#[1,2,4,8,16,32,64]
n_layers = np.max(n_layers_list)           #number of layers for the numerical integration scheme (ResNet)

reversible_mapping = 0 
shared_blocks = 0
backward_order = 1   #Behrmann et al. ICML 2019 used 100
scaling_weights = 1 

output_path = home+'/Sync/Experiments/'+dataset+'/'
prefix = 'gromof'

lambda_x2y = 1
lambda_y2x = 1  
lambda_cycle = 0
lambda_id = 1 
lambda_ae = 1   
lambda_v  = 1 
lambda_Dv = 1 

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
from beo_visu import show_patches

#%%

from beo_nets import feature_model, recon_model, block_mapping_model, forward_block_model
from beo_nets import backward_block_model, mapping_model


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



#%%

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
    
  for l in range(n_layers):
    forward_blocks.append(forward_block)
    backward_blocks.append(backward_block)
    block_x2y_list.append(block_x2y)


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
# from matplotlib.lines import Line2D

# def plot_grad_flow(named_parameters):
#     '''Plots the gradients flowing through different layers in the net during training.
#     Can be used for checking for possible gradient vanishing / exploding problems.
    
#     Usage: Plug this function in Trainer class after loss.backwards() as 
#     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
#     ave_grads = []
#     max_grads= []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#             max_grads.append(p.grad.abs().max())
#     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
#     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(left=0, right=len(ave_grads))
#     #plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
#     plt.ylim(bottom = 0, top=1) # zoom in on the lower gradient regions
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.legend([Line2D([0], [0], color="c", lw=4),
#                 Line2D([0], [0], color="b", lw=4),
#                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
#     print(layers)
    
#%%
# class intermediate_mapping_model(torch.nn.Module):
#   def __init__(self, feature_model, mapping_model):
#     super(intermediate_mapping_model, self).__init__()   
#     self.feature_model = feature_model
#     self.mapping_model = mapping_model

#   def forward(self,x):
#     fx = self.feature_model(x)
#     mx2y = self.mapping_model(fx)

#     return mx2y

#%%

# def power_method_on_jacobian(feature_net,forward_blocks, x, n_iter=5):
#   n = 10
#   u = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)
#   v = torch.Tensor(np.random.rand(1,n_filters,n,n)).to(device)

#   for _ in range (n_iter):
#     WTu = conv_model(torch.flip(torch.flip(u,dims=[2]),dims=[3]))
#     v_new = WTu / torch.norm(WTu.flatten())
#     v = v_new
            
#     Wv = conv_model(v)
#     u_new = Wv / torch.norm(Wv.flatten())
#     u = u_new
              
#   Wv = conv_model(v)
#   return torch.mm( u.flatten().reshape(1,-1), Wv.flatten().reshape(-1,1)).reshape(1)

from beo_lipschitz import block_lipschitz

def loss_sobolev_norm_v(feature_net,forward_blocks, block_x2y_list, n_filters, x):
  norm_velocity = norm_v(feature_net,forward_blocks, x)
  norm_Dv = block_lipschitz(block_x2y_list,n_filters)
  return torch.sum(norm_velocity) + torch.sum(norm_Dv)


# def loss_norm_v(feature_net,forward_blocks, x, along_trajectory = True, only_activation = False, only_gradient = False):
#   norm_velocity, norm_gradient_velocity = norm_inf_v(feature_net,forward_blocks, x, along_trajectory, only_activation, only_gradient)
#   return torch.sum((norm_velocity + norm_gradient_velocity)**2)
  #return torch.max(norm_velocity) + torch.max(norm_gradient_velocity)

def norm_v(feature_net,forward_blocks, x):
  n = len(forward_blocks)
  norm_velocity = torch.zeros([n]).to(device)

  var_x = torch.autograd.Variable(x, requires_grad=True)  
  fx = feature_net(var_x)
  
  for l in range(n):  
    mx = forward_blocks[l](fx)
    vx = mx - fx
    #Max of the infinity norm of vx, which is the incremental update at each forward block
    norm_velocity[l] = torch.max( torch.norm( torch.reshape(vx,(mx.shape[0],-1)), p=np.inf, dim=1 ) )    
    fx = mx
    
  return norm_velocity

# def norm_inf_v(feature_net,forward_blocks, x, along_trajectory = True, only_activation = True, only_gradient = False):

#   n = len(forward_blocks)
#   norm_velocity = torch.zeros([n]).to(device)
#   norm_gradient_velocity = torch.zeros([n]).to(device)

#   var_x = torch.autograd.Variable(x, requires_grad=True)  
#   fx = feature_net(var_x)
  
#   for l in range(n):  
#     mx = forward_blocks[l](fx)
    
#     if only_gradient is False:
#       norm_velocity[l] = torch.max( torch.norm( mx.view(mx.shape[0],-1), p=np.inf, dim=1 ) )
    
#     if only_activation is False:
#       gradients = torch.autograd.grad(outputs=mx, inputs=fx,
#                                       grad_outputs=torch.ones(mx.size()).to(device), # ?
#                             create_graph=True, retain_graph=True, only_inputs=True)[0]
      
#       norm_gradient_velocity[l] = torch.max( torch.norm( gradients.view(gradients.shape[0],-1), p=np.inf, dim=1 ) )
    
#     #if sampling along trajectory otherwise compute only over fx
#     if along_trajectory is True:
#       fx = mx
    
#   return norm_velocity, norm_gradient_velocity

#%%
from beo_lipschitz import compute_lipschitz_conv



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
lw['normdv'] = lambda_Dv  #loss weight for norm Dv
print('loss weights')
print(lw)
 
max_epochs = n_epochs * len(n_layers_list)
validation_norm_velocity = np.zeros( (n_layers , max_epochs) )
validation_norm_gradient_velocity = np.zeros( (n_layers, max_epochs) )  
L = np.zeros( (len(block_x2y_list) * len(block_x2y_list[0] ), max_epochs) )
Lmax = []

#%%

feature_net = feature_model(n_channelsX, n_filters).to(device)
recon_net = recon_model(n_filters, n_channelsY).to(device)

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
validation_loss['reversibility2'] = []


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

  
  mx2y_net = mapping_model(forward_blocks[0:n_blocks]).to(device)
  my2x_net = mapping_model(backward_blocks[0:n_blocks]).to(device)

  net = generator_model(feature_net, mx2y_net, my2x_net, recon_net).to(device)

  # if torch.cuda.device_count() > 1:
  #   print("Let's use", torch.cuda.device_count(), "GPUs!")
  #   net = torch.nn.DataParallel(net)
  #   feature_net = torch.nn.DataParallel(feature_net)
  #   mx2y_net = torch.nn.DataParallel(mx2y_net)
  #   my2x_net = torch.nn.DataParallel(my2x_net)
  #   recon_net = torch.nn.DataParallel(recon_net)
  #   batch_size *= torch.cuda.device_count()

  print('feature net    -- number of trainable parameters : '+str( sum(p.numel() for p in feature_net.parameters() if p.requires_grad) ))
  print('mapping x to y -- number of trainable parameters : '+str( sum(p.numel() for p in mx2y_net.parameters() if p.requires_grad) ))
  print('mapping y to x -- number of trainable parameters : '+str( sum(p.numel() for p in my2x_net.parameters() if p.requires_grad) ))
  print('recon net      -- number of trainable parameters : '+str( sum(p.numel() for p in recon_net.parameters() if p.requires_grad) ))
  print('net            -- number of trainable parameters : '+str( sum(p.numel() for p in net.parameters() if p.requires_grad) ))

  mse = torch.nn.functional.mse_loss
  optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

  for epoch in range(n_epochs):
    torch.cuda.empty_cache()
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
              loss[k] = torch.sum( norm_v(feature_net,forward_blocks, x) )
              #loss[k] = loss_sobolev_norm_v(feature_net,forward_blocks, block_x2y_list, n_filters, x)              
            elif k == 'normdv':
              loss[k] = torch.sum( block_lipschitz(block_x2y_list,n_filters) )
            else:
              loss[k] = mse(eval(k),eval(k[-1]))
          else:
            loss[k] = torch.Tensor([0]).to(device)[0]

        total_loss = 0
        for k in loss.keys():
          if lw[k] > 0:
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
    #plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$'], loc='upper right')
    plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$','$\|\| Dv \|\|_{1,\infty}$'], loc='upper right')
    plt.savefig(output_path+prefix+'_loss_training.png',dpi=150, bbox_inches='tight')
    plt.close()

    print('Computing Lipschitz constants')
    i=0
    for blocks in block_x2y_list:
      for block in blocks:  
        cst = 1
        for layername in block.state_dict().keys():
          if 'conv' in layername:
            weights = block.state_dict()[layername] #torch tensor
            tmp = compute_lipschitz_conv(weights, n_iter = 5)
            cst *= tmp
        L[i,epoch] = cst    
        i = i+1
    Lmax.append(np.max(L[:,epoch]))    
    print('Max Lipschitz value : '+str( np.max(L[:,epoch]) ))
    plt.figure()
    plt.plot(Lmax)
    plt.legend(['Max Lipschitz value'])
    plt.savefig(output_path+prefix+'_lipschitzmax.png',dpi=150, bbox_inches='tight')
    plt.close()   

    plt.figure(figsize=(4,4))
    
    for i in range(L.shape[0]):
      plt.plot(L[i,0:epoch+1])
    plt.savefig(output_path+prefix+'_lipschitz.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    im = plt.imshow(L,cmap=plt.cm.gray, 
                 interpolation="nearest",
                 vmin=0,vmax=3)
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_lipschitz_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

    #------------------------------------------------------------------------------------
    print('\n Validation ')
    net.eval()  
    torch.cuda.empty_cache()
    with torch.no_grad():
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

        fy_tmp = feature_net(y)
        my_tmp = my2x_net(fy_tmp)
        mx_tmp = mx2y_net(my_tmp)
        validation_loss['reversibility2'][-1] += mse(fy_tmp,mx_tmp).item() * x.shape[0]
        
        rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y = net(x,y)
        loss = {}
        for k in lw.keys(): #compute only loss whose weight is non zero
          #if k == 'normv':
          #  loss[k] = loss_sobolev_norm_v(feature_net,forward_blocks, block_x2y_list, n_filters, x)#loss_norm_v(feature_net,forward_blocks[0:n_blocks],x)
          if k == 'normv':
            loss[k] = torch.sum( norm_v(feature_net,forward_blocks, x) )
          elif k == 'normdv':
            loss[k] = torch.sum( block_lipschitz(block_x2y_list,n_filters) )
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
      #plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$'], loc='upper right')
      plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$','$\|\| v \|\|_{1,\infty}$','$\|\| Dv \|\|_{1,\infty}$'], loc='upper right')
      plt.savefig(output_path+prefix+'_loss_validation.png',dpi=150, bbox_inches='tight')
      plt.close()
      
      plt.figure()
      plt.plot(validation_loss['reversibility'])
      plt.plot(validation_loss['reversibility2'])
      plt.legend(['$\| f(x) - m^{-1} \circ m \circ f(x) \|^2$','$\| f(y) - m \circ m^{-1} \circ f(y) \|^2$'])
      plt.savefig(output_path+prefix+'_inverr.png',dpi=150, bbox_inches='tight')
      plt.close()   

      print('\nComputing norm velocity...')  
      for (x,y) in testloader:
        x, y = x.to(device), y.to(device)

        nv = norm_v(feature_net,forward_blocks, x)
        ngv = block_lipschitz(block_x2y_list,n_filters)

        for l in range(len(forward_blocks)):
          validation_norm_velocity[l,iteration] = np.maximum(nv[l].cpu().detach().numpy(), validation_norm_velocity[l,iteration])
          validation_norm_gradient_velocity[l,iteration] = np.maximum(ngv[l].cpu().detach().numpy(), validation_norm_gradient_velocity[l,iteration])
                      
      plt.figure(figsize=(4,4))
      
      for i in range(validation_norm_velocity.shape[0]):
        plt.plot(validation_norm_velocity[i,0:iteration+1])
      plt.savefig(output_path+prefix+'_norm_velocity.png',dpi=150, bbox_inches='tight')
      plt.close()

      plt.figure(figsize=(4,4))    
      im = plt.imshow(validation_norm_velocity,cmap=plt.cm.gray,
                  interpolation="nearest")
      plt.colorbar(im)
      plt.savefig(output_path+prefix+'_norm_velocity_matrix.png',dpi=150, bbox_inches='tight')
      plt.close()

      plt.figure(figsize=(4,4))
      
      for i in range(validation_norm_gradient_velocity.shape[0]):
        plt.plot(validation_norm_gradient_velocity[i,0:iteration+1])
      plt.savefig(output_path+prefix+'_norm_gradient_velocity.png',dpi=150, bbox_inches='tight')
      plt.close()

      plt.figure(figsize=(4,4))
      
      im = plt.imshow(validation_norm_gradient_velocity,cmap=plt.cm.gray,
                  interpolation="nearest")
      plt.colorbar(im)
      plt.savefig(output_path+prefix+'_norm_gradient_velocity_matrix.png',dpi=150, bbox_inches='tight')
      plt.close()

      print('max norm velocity : '+str(np.max(validation_norm_velocity[:,iteration])))
      print('max norm gradient velocity : '+str(np.max(validation_norm_gradient_velocity[:,iteration])))
    
    iteration += 1 
    
print('last iteration : '+str(iteration))

#%%
#Save everything in a dictionary

d = {}
d['prefix'] = prefix
d['forward_blocks'] = forward_blocks
d['backward_blocks'] = backward_blocks
d['block_x2y_list'] = block_x2y_list
d['feature_net'] = feature_net
d['mx2y_net'] = mx2y_net
d['my2x_net'] = my2x_net
d['recon_net'] = recon_net
d['net'] = net

d['training_loss'] = training_loss
d['validation_loss'] = validation_loss
d['validation_norm_velocity'] = validation_norm_velocity
d['validation_norm_gradient_velocity'] = validation_norm_gradient_velocity
d['L'] = L
d['Lmax'] = Lmax

joblib.dump(d,output_path+prefix+'_dictionary.pkl', compress=True)



#%%



#torch.save(net,output_path+prefix+'_net')