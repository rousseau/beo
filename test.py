#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:52:14 2019

@author: rousseau
"""

#%%
import joblib
import numpy as np
import keras.backend as K
from model_zoo import build_feature_model_2d, build_recon_model_2d, build_block_model_2d
from model_zoo import build_encoder_2d, build_decoder_2d
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition, build_model
from model_zoo import build_exp_model, fm_model, build_reversible_forward_model_2d, build_reversible_backward_model_2d, build_4_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import nibabel
from dataset import get_ixi_2dpatches, get_hcp_2dpatches
import subprocess
import socket

#%%
 
def show_patches(patch_list,filename):

  n_rows = patch_list[0].shape[0]
  n_cols = len(patch_list)

  vmin = -2 
  vmax = 2
  plt.figure(figsize=(2. * n_cols, 2. * n_rows))
  for i in range(n_rows):
    for j in range(n_cols):
      sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
      sub.imshow(patch_list[j][i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest",
                 vmin=vmin,vmax=vmax)
      sub.axis('off')
  plt.axis('off')
  plt.savefig(filename,dpi=150, bbox_inches='tight')
  plt.close()




print(K.image_data_format())
from os.path import expanduser
home = expanduser("~")
dataset = 'HCP'
output_path = home+'/Sync/Experiments/'+dataset+'/'

patch_size = 80 
load_pickle_patches = 1

if load_pickle_patches == 0:

  if dataset == 'IXI':  
    n_patches = 100 #per slice 
    (T1_2D,T2_2D,PD_2D) = get_ixi_2dpatches(patch_size = patch_size, n_patches = n_patches)
    joblib.dump(T1_2D,home+'/Exp/T1_2D.pkl', compress=True)
    joblib.dump(T2_2D,home+'/Exp/T2_2D.pkl', compress=True)
    joblib.dump(PD_2D,home+'/Exp/PD_2D.pkl', compress=True)
  if dataset == 'HCP':  
    n_patches = 50 #per slice 
    (T1_2D,T2_2D) = get_hcp_2dpatches(patch_size = patch_size, n_patches = n_patches)
    joblib.dump(T1_2D,home+'/Exp/HCP_T1_2D.pkl', compress=True)
    joblib.dump(T2_2D,home+'/Exp/HCP_T2_2D.pkl', compress=True)
  
else:  
  print('Loading gzip pickle files')  
  if dataset == 'IXI':
    T1_2D = joblib.load(home+'/Exp/T1_2D.pkl')
    T2_2D = joblib.load(home+'/Exp/T2_2D.pkl')
    PD_2D = joblib.load(home+'/Exp/PD_2D.pkl')
  if dataset == 'HCP':  
    T1_2D = joblib.load(home+'/Exp/HCP_T1_2D.pkl')
    T2_2D = joblib.load(home+'/Exp/HCP_T2_2D.pkl')
    
  print(T1_2D.shape)  

#%%
  
n_channelsX = 1
n_channelsY = 1
n_filters = 32
n_layers = 2 
n_layers_residual = 2
learning_rate = 0.00001
loss = 'mae' 
batch_size = 32     
epochs = 10    
use_optim = 0
kernel_size = 5
shared_blocks = 0
reversible = 0 
f_shared = 1  
r_shared = 1
n_gpu = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID') #Get automatically the number of GPUs : https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
encdec = 1
n_levels = 1 #Check for 0 level, there is a bug...  


if K.image_data_format() == 'channels_first':
  init_shape = (n_channelsX, None, None)
  feature_shape = (n_filters, None, None)
  half_feature_shape = ((int)(n_filters/2), None, None)
  channel_axis = 1
else:
  init_shape = (None, None,n_channelsX)
  feature_shape = (None, None, n_filters)
  half_feature_shape = (None, None, (int)(n_filters/2))
  channel_axis = -1  
  
#%%

if encdec == 0:
  fm_x2y = build_feature_model_2d(init_shape=init_shape, n_filters=n_filters, kernel_size=kernel_size)
  fm_y2x = build_feature_model_2d(init_shape=init_shape, n_filters=n_filters, kernel_size=kernel_size)
  fm = build_feature_model_2d(init_shape=init_shape, n_filters=n_filters, kernel_size=kernel_size)


  rm_x2y = build_recon_model_2d(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=kernel_size)
  rm_y2x = build_recon_model_2d(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=kernel_size)
  rm = build_recon_model_2d(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=kernel_size)

else:

  fm_x2y = build_encoder_2d(init_shape=init_shape, n_filters=n_filters, n_levels = n_levels)
  fm_y2x = build_encoder_2d(init_shape=init_shape, n_filters=n_filters, n_levels = n_levels)
  fm = build_encoder_2d(init_shape=init_shape, n_filters=n_filters, n_levels = n_levels)


  rm_x2y = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = n_levels)
  rm_y2x = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = n_levels)
  rm = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = n_levels)


if reversible == 1:  
  block_f_x2y = []
  block_g_x2y = []
  if shared_blocks == 0:
    for l in range(n_layers):
      block_f_x2y.append(build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual))
      block_g_x2y.append(build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual))
  else:
    bf = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)
    bg = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)
    for l in range(n_layers):
      block_f_x2y.append(bf)
      block_g_x2y.append(bg)

  mm_x2y = build_reversible_forward_model_2d(init_shape=feature_shape, block_f=block_f_x2y, block_g = block_g_x2y, n_layers=n_layers)
  mm_y2x = build_reversible_backward_model_2d(init_shape=feature_shape, block_f=block_f_x2y, block_g = block_g_x2y, n_layers=n_layers)

else:
  block_x2y = []
  if shared_blocks == 0:
    for l in range(n_layers):
      block_x2y.append( build_block_model_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual) )
  else:
    b = build_block_model_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual)
    for l in range(n_layers):
      block_x2y.append(b)

  mm_x2y = build_forward_model(init_shape=feature_shape, block_model=block_x2y, n_layers=n_layers)
  mm_y2x = build_backward_model(init_shape=feature_shape, block_model=block_x2y, n_layers=n_layers)


#%%
from keras.models import Model
from keras.layers import Input

ix = Input(shape=init_shape)
iy = Input(shape=init_shape)

if f_shared == 1:
  fx = fm(ix) # features of x
  fy = fm(iy) # features of y
else:
  fx = fm_x2y(ix) # features of x
  fy = fm_y2x(iy) # features of y

mx2y = mm_x2y(fx) # direct mapping
my2x = mm_y2x(fy) # inverse mapping

if r_shared == 1:
  rx2y = rm(mx2y) # reconstruction of direct mapping
  ry2x = rm(my2x) # reconstruction of inverse mapping
else:
  rx2y = rm_x2y(mx2y) # reconstruction of direct mapping
  ry2x = rm_y2x(my2x) # reconstruction of inverse mapping

model = Model(inputs=[ix,iy], outputs=[rx2y,ry2x])

model_x2y = Model(inputs=ix, outputs=rx2y)
model_y2x = Model(inputs=iy, outputs=ry2x)

print('f_shared : '+str(f_shared))
print('r_shared : '+str(r_shared))
print('reversible : '+str(reversible))
print('batch_size : '+str(batch_size))
print('n_filters : '+str(n_filters))
print('n_gpu : '+str(n_gpu))

info = '_gpu'+str(n_gpu)+'_bs'+str(batch_size)+'_nf'+str(n_filters)+'_f'+str(f_shared)+'_r'+str(r_shared)+'_rev'+str(reversible)+'_enc'+str(encdec)+'_nlevels'+str(n_levels)

#%%

#Select the model to optimze wrt.
if n_gpu > 1:
  model_gpu = multi_gpu_model(model, gpus=n_gpu)
  model_x2y_gpu = multi_gpu_model(model_x2y, gpus=n_gpu)
  model_y2x_gpu = multi_gpu_model(model_y2x, gpus=n_gpu)  
  batch_size = batch_size * n_gpu
else:
  model_x2y_gpu = model_x2y
  model_y2x_gpu = model_y2x
  model_gpu = model


model_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss)
model_x2y_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss)
model_y2x_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss)

model_gpu.summary()


#%%
n = 10
step = 10000
t2 = T2_2D[0:n*step:step,:,:,:]
t1 = T1_2D[0:n*step:step,:,:,:]

loss_t2 = []
loss_t1 = []
loss_keras=[]
for epoch in range(epochs):

  # if epoch == 0:
  #   batch_size = 16
  # if epoch == 5:
  #   batch_size = 32
  # if epoch == 10:
  #   batch_size = 64
  # if epoch == 15:
  #   batch_size = 128 
    
    
  hist = model_gpu.fit(x=[T2_2D,T1_2D], y=[T1_2D,T2_2D], batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
  loss_keras.append(hist.history['loss'])
  #model_x2y_gpu.fit(x=T2_2D, y=T1_2D, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

  a = model_x2y_gpu.predict(x=t2, batch_size = batch_size)
  print('keras loss : '+str(model_x2y_gpu.evaluate(x=t2, y=t1)))
  print('mae synthesis t1 : '+str(np.mean(np.abs(t1 - a))))
  loss_t1.append(np.mean(np.abs(t1 - a)))
  #model_y2x_gpu.fit(x=T1_2D, y=T2_2D, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)  

  b = model_y2x_gpu.predict(x=t1, batch_size = batch_size)
  print('keras loss : '+str(model_y2x_gpu.evaluate(x=t1, y=t2)))  
  print('mae synthesis t2 : '+str(np.mean(np.abs(t2 - b))))
  loss_t2.append(np.mean(np.abs(t2 - b)))

  [a,b] = model_gpu.predict(x=[t2,t1], batch_size = batch_size) 
  print('keras loss : ')
  print(model_gpu.evaluate(x=[t2,t1], y=[t1,t2]))    
  print('mae synthesis t1 : '+str(np.mean(np.abs(t1 - a))))
  print('mae synthesis t2 : '+str(np.mean(np.abs(t2 - b))))
 
  show_patches(patch_list=[t2,t1,a,b],filename=output_path+socket.gethostname()+info+'_e'+str(epoch)+'fig_patches.png')

  plt.figure()
  plt.plot(loss_t1)
  plt.plot(loss_t2)
  plt.plot(loss_keras)
  plt.legend(['T1', 'T2', 'keras'])
  plt.savefig(output_path+socket.gethostname()+info+'_loss.png',dpi=150, bbox_inches='tight')
  plt.close()

#%%

 #Check if the model is reversible

x = Input(shape=init_shape)	
fx= fm_x2y(x)
mx= mm_x2y(fx)
my= mm_y2x(mx)
tmpmodel = Model(inputs=x, outputs=[fx,my])
tmpmodel.compile(optimizer=Adam(lr=learning_rate), 
               loss=loss)

[a,b] = tmpmodel.predict(x=T2_2D[0:n*step:step,:,:,:], batch_size = batch_size)
print('Error on reversible mapping : '+str(np.linalg.norm(a-b))) 










  