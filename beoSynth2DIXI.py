#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:18:06 2019

@author: rousseau
"""

#%%
import joblib
import numpy as np
import keras.backend as K
from model_zoo import build_feature_model_2d, build_recon_model_2d, build_block_model_2d
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition
from model_zoo import build_exp_model, fm_model, build_reversible_forward_model_2d, build_reversible_backward_model_2d, build_4_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import nibabel
from dataset import get_ixi_2dpatches
import subprocess

#%%


print(K.image_data_format())
from os.path import expanduser
home = expanduser("~")
output_path = home+'/Sync/Experiments/IXI/'

patch_size = 80
load_pickle_patches = 1

if load_pickle_patches == 0:
  n_patches = 50 #per slice
  (T1_2D,T2_2D,PD_2D) = get_ixi_2dpatches(patch_size = patch_size, n_patches = n_patches)
  joblib.dump(T1_2D,home+'/Exp/T1_2D.pkl', compress=True)
  joblib.dump(T2_2D,home+'/Exp/T2_2D.pkl', compress=True)
  joblib.dump(PD_2D,home+'/Exp/PD_2D.pkl', compress=True)
  
else:  
  print('Loading gzip pickle files')  
  T1_2D = joblib.load(home+'/Exp/T1_2D.pkl')
  T2_2D = joblib.load(home+'/Exp/T2_2D.pkl')
  PD_2D = joblib.load(home+'/Exp/PD_2D.pkl')
  
  print(T1_2D.shape)


#plt.figure()  
#
#for i in range(5):
#  sub = plt.subplot(1, 3, 1)
#  sub.imshow(T1_2D[i*10,:,:,0], cmap=plt.cm.gray, interpolation="nearest")
#  sub = plt.subplot(1, 3, 2)
#  sub.imshow(T2_2D[i*10,:,:,0], cmap=plt.cm.gray, interpolation="nearest")
#  sub = plt.subplot(1, 3, 3)
#  sub.imshow(PD_2D[i*10,:,:,0], cmap=plt.cm.gray, interpolation="nearest")
#
#  plt.show(block=False)    
#%%
  

n_channelsX = 1
n_channelsY = 1
n_filters = 32
n_layers = 1
n_layers_residual = 5
learning_rate = 0.0001
loss = 'mae' 
batch_size = 128 
epochs = 15
use_optim = 0
kernel_size = 5
shared_blocks = 0
#Get automatically the number of GPUs : https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
n_gpu = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')

lw1 = 1 #loss weight direct mapping
lw2 = 1 #loss weight inverse mapping
lw3 = 0 #loss weight identity mapping

inverse_consistency = 0
cycle_consistency = 0
identity_consistency = 0

prefix = 'iclr_nowindowing'
if use_optim == 1:
  prefix += '_optim'
if inverse_consistency == 1:
  prefix += '_invc'
if cycle_consistency == 1:
  prefix += '_cycle'
if identity_consistency == 1:
  prefix += '_idc'  
if shared_blocks == 1:
  prefix += '_shared'  
prefix+= '_e'+str(epochs)+'_ps'+str(patch_size)+'_np'+str(T1_2D.shape[0])
prefix+= '_bs'+str(batch_size)
prefix+= '_lr'+str(learning_rate)
prefix+= '_nl'+str(n_layers)
prefix+= '_nlr'+str(n_layers_residual)
prefix+= '_lw'+str(lw1)+'_'+str(lw2)+'_'+str(lw3)
prefix+= '_'

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
      
feature_model = build_feature_model_2d(init_shape=init_shape, n_filters=n_filters, kernel_size=kernel_size)
recon_model = build_recon_model_2d(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=kernel_size)

#block_f_T2_to_T1 = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)
#block_g_T2_to_T1 = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)

block_f_T2_to_T1 = []
block_g_T2_to_T1 = []
if shared_blocks == 0:
  for l in range(n_layers):
    block_f_T2_to_T1.append(build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual))
    block_g_T2_to_T1.append(build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual))
else:
  bf = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)
  bg = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)
  for l in range(n_layers):
    block_f_T2_to_T1.append(bf)
    block_g_T2_to_T1.append(bg)
  
mapping_reversible_T2_to_T1 = build_reversible_forward_model_2d(init_shape=feature_shape, block_f=block_f_T2_to_T1, block_g = block_g_T2_to_T1, n_layers=n_layers)
mapping_reversible_T1_to_T2 = build_reversible_backward_model_2d(init_shape=feature_shape, block_f=block_f_T2_to_T1, block_g = block_g_T2_to_T1, n_layers=n_layers)

model_all = build_4_model(init_shape, feature_model, mapping_reversible_T2_to_T1, mapping_reversible_T1_to_T2, recon_model)

if n_gpu > 1:
  model = multi_gpu_model(model_all, gpus=n_gpu)
  batch_size = batch_size * n_gpu
else:
  model = model_all
  

model.compile(optimizer=Adam(lr=learning_rate), 
                loss=loss,
                loss_weights=[lw1,lw2,lw3,lw3])
model.summary()

#%%
for e in range(epochs):
  lr = learning_rate
  if e > 5:  
    lr *= 0.1
  elif e > 10:
    lr *= 0.1
  
  print('current learning rate : '+str(lr))
  
  model.compile(optimizer=Adam(lr=lr), 
                  loss=loss,
                  loss_weights=[lw1,lw2,lw3,lw3])
  
  
  model.fit(x=[T2_2D,T1_2D], y=[T1_2D,T2_2D,T2_2D,T1_2D], batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
  
  n = 10
  step = 100
  [a,b,c,d] = model.predict(x=[T2_2D[0:n*step:step,:,:,:],T1_2D[0:n*step:step,:,:,:]], batch_size = batch_size)
  t2 = T2_2D[0:n*step:step,:,:,:]
  t1 = T1_2D[0:n*step:step,:,:,:]
  
  n_cols = 6
  n_rows = n
  #index = np.random.permutation(T1.shape[0])
  #
  plt.figure(figsize=(2. * n_cols, 2. * n_rows))
  for i in range(n):
    sub = plt.subplot(n_rows, n_cols, i*n_cols+1)
    sub.imshow(t2[i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest")
    sub = plt.subplot(n_rows, n_cols, i*n_cols+2)
    sub.imshow(t1[i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest")
    sub = plt.subplot(n_rows, n_cols, i*n_cols+3)  
    sub.imshow(a[i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest")
    sub = plt.subplot(n_rows, n_cols, i*n_cols+4)  
    sub.imshow(b[i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest")
    sub = plt.subplot(n_rows, n_cols, i*n_cols+5)  
    sub.imshow(c[i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest")
    sub = plt.subplot(n_rows, n_cols, i*n_cols+6)  
    sub.imshow(d[i,:,:,0],
                 cmap=plt.cm.gray,
                 interpolation="nearest")
    
  #plt.show()
  plt.axis('off')
  plt.savefig(output_path+prefix+'_current'+str(e)+'fig_patches.png',dpi=150, bbox_inches='tight')
  plt.close()

#%%
  #Check if the model is reversible
from keras.layers import Input
from keras.models import Model

x = Input(shape=init_shape)	
fx= feature_model(x)
mx= mapping_reversible_T2_to_T1(fx)
my= mapping_reversible_T1_to_T2(mx)
tmpmodel = Model(inputs=x, outputs=[fx,my])
tmpmodel.compile(optimizer=Adam(lr=learning_rate), 
                  loss=loss)

[a,b] = tmpmodel.predict(x=T2_2D[0:n*step:step,:,:,:], batch_size = batch_size)
print(np.linalg.norm(a-b))  
#%%  

#Apply on a test image
T1image = nibabel.load(output_path+'IXI661-HH-2788-T1_N4.nii.gz')
T2image = nibabel.load(output_path+'IXI661-HH-2788-T2_N4.nii.gz')
PDimage = nibabel.load(output_path+'IXI661-HH-2788-PD_N4.nii.gz')
maskarray = nibabel.load(output_path+'IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)
