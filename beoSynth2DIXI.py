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
from model_zoo import build_encoder_2d, build_decoder_2d
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition, build_model
from model_zoo import build_exp_model, fm_model, build_reversible_forward_model_2d, build_reversible_backward_model_2d, build_4_model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from dataset import get_ixi_2dpatches, get_hcp_2dpatches, get_hcp_4darrays
from sklearn.model_selection import train_test_split
import subprocess
import socket
from keras.layers import Input
from keras.models import Model

from os.path import expanduser
home = expanduser("~")

#import sys
#sys.path.append(home+'/Sync/Code/keras_one_cycle_clr-master/')

#todo list
#bien separer les donnees training and testing (fait pour hcp)
#visualiser les normes du velocity field (check approximation de l'inverse)
#save the models and run predictions on whole testing image
#add data augmentation
#1cycle optimization


#%%
 
def show_patches(patch_list,filename):

  titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$']
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
      if i == 0:
        sub.set_title(titles[j])
  plt.axis('off')
  plt.savefig(filename,dpi=150, bbox_inches='tight')
  plt.close()


#%%
print(K.image_data_format())


dataset = 'HCP' #HCP or IXI
output_path = home+'/Sync/Experiments/'+dataset+'/'

patch_size = 80 
load_pickle_patches = 1
ratio_training = 0.75

if load_pickle_patches == 0:

  if dataset == 'IXI':  
    n_patches = 100 #per slice 
    (T1_2D,T2_2D,PD_2D) = get_ixi_2dpatches(patch_size = patch_size, n_patches = n_patches)
    joblib.dump(T1_2D,home+'/Exp/T1_2D.pkl', compress=True)
    joblib.dump(T2_2D,home+'/Exp/T2_2D.pkl', compress=True)
    joblib.dump(PD_2D,home+'/Exp/PD_2D.pkl', compress=True)
  if dataset == 'HCP':  
    n_patches = 50 #per slice 
    
    (T1s,T2s,masks) = get_hcp_4darrays()
    n_images = len(T1s)
    print('Number of loaded images : '+str(n_images))
    n_training_images = (int)(ratio_training*n_images)
    print('Number of training images : '+str(n_training_images))
    
    T1_training = T1s[0:n_training_images]
    T2_training = T2s[0:(int)(ratio_training*n_images)]
    mask_training = masks[0:(int)(ratio_training*n_images)]    
    
    (T1_2D_training,T2_2D_training) = get_hcp_2dpatches(patch_size = patch_size, n_patches = n_patches, data=(T1s[0:n_training_images],T2s[0:n_training_images],masks[0:n_training_images]))    
    joblib.dump(T1_2D_training,home+'/Exp/HCP_T1_2D_training.pkl', compress=True)
    joblib.dump(T2_2D_training,home+'/Exp/HCP_T2_2D_training.pkl', compress=True)

    (T1_2D_testing,T2_2D_testing) = get_hcp_2dpatches(patch_size = patch_size, n_patches = n_patches, data=(T1s[n_training_images:n_images],T2s[n_training_images:n_images],masks[n_training_images:n_images]))    
    joblib.dump(T1_2D_testing,home+'/Exp/HCP_T1_2D_testing.pkl', compress=True)
    joblib.dump(T2_2D_testing,home+'/Exp/HCP_T2_2D_testing.pkl', compress=True)
  
else:  
  print('Loading gzip pickle files')  
  if dataset == 'IXI':
    T1_2D = joblib.load(home+'/Exp/T1_2D.pkl')
    T2_2D = joblib.load(home+'/Exp/T2_2D.pkl')
    PD_2D = joblib.load(home+'/Exp/PD_2D.pkl')
  if dataset == 'HCP':  
    T1_2D_training = joblib.load(home+'/Exp/HCP_T1_2D_training.pkl')
    T2_2D_training = joblib.load(home+'/Exp/HCP_T2_2D_training.pkl')
    T1_2D_testing = joblib.load(home+'/Exp/HCP_T1_2D_testing.pkl')
    T2_2D_testing = joblib.load(home+'/Exp/HCP_T2_2D_testing.pkl')
    
#%%
x_train = T2_2D_training
x_test  = T2_2D_testing

y_train = T1_2D_training
y_test  = T1_2D_testing


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
n_layers = 5           #number of layers for the numerical integration scheme (ResNet)
n_layers_residual = 1  #number of layers for the parametrization of the velocity fields
block_type = 'bottleneck'
learning_rate = 0.001 #changer le lr ? et l'adapter en fonction des epochs?
loss = 'mae' 
batch_size = 32  
epochs = 100
use_optim = 0
kernel_size = 5
shared_blocks = 0  
#Get automatically the number of GPUs : https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
n_gpu = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
 
lw1 = 1 #loss weight direct mapping
lw2 = 1 #loss weight inverse mapping
lw3 = 1 #loss weight identity mapping on x
lw4 = 1 #loss weight identity mapping on y
lw5 = 0 #loss weight cycle for x
lw6 = 0 #loss weight cycle for y
lw7 = 1 #loss weight autoencoder for x
lw8 = 1 #loss weight autoencoder for y

lws = [lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8]
   
inverse_consistency = 0
cycle_consistency = 0
identity_consistency = 0
reversible = 1 
encdec = 1
n_levels = 1 #Check for 0 level, there is a bug...  
  

prefix = socket.gethostname()+'_'+dataset
if use_optim == 1:
  prefix += '_optim'
if inverse_consistency == 1:
  prefix += '_invc'
if cycle_consistency == 1:
  prefix += '_cycle'
if identity_consistency == 1:
  prefix += '_idc'  
if reversible == 1:
  prefix += '_rev'    
if shared_blocks == 1:
  prefix += '_shared'  
prefix+= '_e'+str(epochs)+'_ps'+str(patch_size)+'_np'+str(x_train.shape[0])
prefix+= '_bs'+str(batch_size)
prefix+= '_lr'+str(learning_rate)
prefix+= '_nl'+str(n_layers)
prefix+= '_nlr'+str(n_layers_residual)
prefix+= '_'+block_type
prefix+= '_lw'+str(lw1)+'_'+str(lw2)+'_'+str(lw3)+'_'+str(lw4)+'_'+str(lw5)+'_'+str(lw6)+'_'+str(lw7)+'_'+str(lw8)
prefix+= '_encdec'+str(encdec)
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

if encdec == 0:      
  feature_model = build_feature_model_2d(init_shape=init_shape, n_filters=n_filters, kernel_size=kernel_size)
  recon_model = build_recon_model_2d(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=kernel_size)
else:
  feature_model = build_encoder_2d(init_shape=init_shape, n_filters=n_filters, n_levels = n_levels)
  recon_model = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = n_levels)
  

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

  mapping_x2y = build_reversible_forward_model_2d(init_shape=feature_shape, block_f=block_f_x2y, block_g = block_g_x2y, n_layers=n_layers)
  mapping_y2x = build_reversible_backward_model_2d(init_shape=feature_shape, block_f=block_f_x2y, block_g = block_g_x2y, n_layers=n_layers)

else:
  block_x2y = []
  if shared_blocks == 0:
    for l in range(n_layers):
      block_x2y.append( build_block_model_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual) )
  else:
    b = build_block_model_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual)
    for l in range(n_layers):
      block_x2y.append(b)

  mapping_x2y = build_forward_model(init_shape=feature_shape, block_model=block_x2y, n_layers=n_layers)
  mapping_y2x = build_backward_model(init_shape=feature_shape, block_model=block_x2y, n_layers=n_layers)
  

model_all = build_4_model(init_shape, feature_model, mapping_x2y, mapping_y2x, recon_model)

#Select the model to optimze wrt.
if n_gpu > 1:
  model_all_gpu = multi_gpu_model(model_all, gpus=n_gpu)
  batch_size = batch_size * n_gpu
else:
  model_all_gpu = model_all
  
model_all_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss, loss_weights=lws)

#print('The All model:')
#model_all_gpu.summary()

print('Number of parameters of feature_model model : '+str(feature_model.count_params()))
if reversible == 1:
  print('Number of parameters of block_f_x2y model : '+str(block_f_x2y[0].count_params()))
  print('Number of parameters of block_g_x2y model : '+str(block_g_x2y[0].count_params()))
else:
  print('Number of parameters of block_x2y model : '+str(block_x2y[0].count_params()))  
print('Number of parameters of mapping_x2y model : '+str(mapping_x2y.count_params()))
print('Number of parameters of recon_model model : '+str(recon_model.count_params()))
print('Number of parameters of model_all_gpu model : '+str(model_all_gpu.count_params()))

#%%
#From  : https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras/notebook
import keras

class CyclicLR(keras.callbacks.Callback):
    
  def __init__(self,base_lr, max_lr, step_size, base_m, max_m, cyclical_momentum):

    self.base_lr = base_lr
    self.max_lr = max_lr
    self.base_m = base_m
    self.max_m = max_m
    self.cyclical_momentum = cyclical_momentum
    self.step_size = step_size
    
    self.clr_iterations = 0.
    self.cm_iterations = 0.
    self.trn_iterations = 0.
    self.history = {}
      
  def clr(self):
      
    cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
    
    if cycle == 2:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)          
        return self.base_lr-(self.base_lr-self.base_lr/100)*np.maximum(0,(1-x))
    
    else:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0,(1-x))
  
  def cm(self):
      
    cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
    
    if cycle == 2:
        
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1) 
        return self.max_m
    
    else:
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        return self.max_m - (self.max_m-self.base_m)*np.maximum(0,(1-x))
      
      
  def on_train_begin(self, logs={}):
    logs = logs or {}

    if self.clr_iterations == 0:
        K.set_value(self.model.optimizer.lr, self.base_lr)
    else:
        K.set_value(self.model.optimizer.lr, self.clr())
        
    if self.cyclical_momentum == True:
        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.momentum, self.cm())
        else:
            K.set_value(self.model.optimizer.momentum, self.cm())
          
          
  def on_batch_begin(self, batch, logs=None):
      
    logs = logs or {}
    self.trn_iterations += 1
    self.clr_iterations += 1

    self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
    self.history.setdefault('iterations', []).append(self.trn_iterations)
    
    if self.cyclical_momentum == True:
        self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))

    for k, v in logs.items():
        self.history.setdefault(k, []).append(v)
    
    K.set_value(self.model.optimizer.lr, self.clr())
    
    if self.cyclical_momentum == True:
        K.set_value(self.model.optimizer.momentum, self.cm())

max_lr = 0.0005
base_lr = max_lr/10
max_m = 0.98
base_m = 0.85

cyclical_momentum = False
augment = True
cycles = 2.35
epochs=10

iterations = round(x_train.shape[0]/batch_size*epochs)
iterations = list(range(0,iterations+1))
step_size = len(iterations)/(cycles)

clr =  CyclicLR(base_lr=base_lr,
                max_lr=max_lr,
                step_size=step_size,
                max_m=max_m,
                base_m=base_m,
                cyclical_momentum=cyclical_momentum)

callbacks = [clr]
history = model_all_gpu.fit(x=[x_train,y_train], 
                          y=[y_train,x_train,x_train,y_train,x_train,y_train,x_train,y_train], 
                          batch_size=batch_size, 
                          epochs=10, 
                          verbose=1, 
                          shuffle=True,
                          callbacks=callbacks,
                          validation_data=([x_test,y_test],[y_test,x_test,x_test,y_test,x_test,y_test,x_test,y_test]))

plt.plot(clr.history['iterations'], clr.history['lr'])
plt.xlabel('Training Iterations')
plt.ylabel('Learning Rate')
plt.title("One Cycle Policy")
plt.show()

plt.plot(clr.history['iterations'], clr.history['momentum'])
plt.xlabel('Training Iterations')
plt.ylabel('Momentum')
plt.title("One Cycle Policy")
plt.show()

#%%
val_loss = history.history['val_loss']
loss = history.history['loss']
plt.plot(range(len(val_loss)),val_loss,'c',label='Validation loss')
plt.plot(range(len(loss)),loss,'m',label='Train loss')

plt.title('Training and validation losses')
plt.legend()
plt.xlabel('epochs')
plt.show()

#%%

keras_val = None
keras_loss = None
losses = None
psnrs  = None
inverr = []

for epoch in range(epochs):
  hist = model_all_gpu.fit(x=[x_train,y_train], 
                           y=[y_train,x_train,x_train,y_train,x_train,y_train,x_train,y_train], 
                           batch_size=batch_size, 
                           epochs=1, 
                           verbose=1, 
                           shuffle=True,
                           validation_data=([x_test,y_test],[y_test,x_test,x_test,y_test,x_test,y_test,x_test,y_test]))

  n = 10
  step = 4000
  t2 = x_test[0:n*step:step,:,:,:]
  t1 = y_test[0:n*step:step,:,:,:]
  [a,b,c,d,e,f,g,h] = model_all_gpu.predict(x=[t2,t1], batch_size = batch_size)
 
  res = np.zeros((1,8))
  res[0,0] = np.mean((t1 - a)**2)
  res[0,1] = np.mean((t2 - b)**2)
  res[0,2] = np.mean((t2 - c)**2)
  res[0,3] = np.mean((t1 - d)**2)
  res[0,4] = np.mean((t2 - e)**2)
  res[0,5] = np.mean((t1 - f)**2)
  res[0,6] = np.mean((t2 - g)**2)
  res[0,7] = np.mean((t1 - h)**2)

  if losses is None:
    losses = np.copy(res)
  else:
    losses = np.concatenate((losses,res),axis=0)  

  plt.figure()
  plt.plot(losses)
  plt.xlabel('epochs')
  plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
  plt.savefig(output_path+prefix+'_losses.png',dpi=150, bbox_inches='tight')
  plt.close()
     
  show_patches(patch_list=[t2,t1,a,b,c,d,e,f,g,h],filename=output_path+prefix+'_current'+str(epoch)+'fig_patches.png')
  
  dyn_t1 = np.max(t1) - np.min(t1)
  dyn_t2 = np.max(t2) - np.min(t2)
  
  psnr = np.zeros((1,8))
  psnr[0,0]= 10*np.log10( dyn_t1 / (np.mean((t1 - a)**2) ))
  psnr[0,1]= 10*np.log10( dyn_t2 / (np.mean((t2 - b)**2) ))  
  psnr[0,2]= 10*np.log10( dyn_t2 / (np.mean((t2 - c)**2) ))  
  psnr[0,3]= 10*np.log10( dyn_t1 / (np.mean((t1 - d)**2) ))  
  psnr[0,4]= 10*np.log10( dyn_t2 / (np.mean((t2 - e)**2) )) 
  psnr[0,5]= 10*np.log10( dyn_t1 / (np.mean((t1 - f)**2) ))  
  psnr[0,6]= 10*np.log10( dyn_t2 / (np.mean((t2 - g)**2) ))  
  psnr[0,7]= 10*np.log10( dyn_t1 / (np.mean((t1 - h)**2) ))    
  
  if psnrs is None:
    psnrs = np.copy(psnr)
  else:
    psnrs = np.concatenate((psnrs,psnr),axis=0)  
      
  plt.figure()
  plt.plot(psnrs)
  plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
  plt.savefig(output_path+prefix+'_psnr.png',dpi=150, bbox_inches='tight')
  plt.close()  
  
  val_loss = np.zeros((1,8))
  for i in range(8):
    val_loss[0,i] = hist.history['val_model_2_'+str(i+1)+'_loss'][0]
  if keras_val is None:
    keras_val = np.copy(val_loss)
  else:
    keras_val = np.concatenate((keras_val,val_loss),axis=0)  
  plt.figure()
  plt.plot(keras_val)
  plt.xlabel('epochs')
  plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
  plt.savefig(output_path+prefix+'_keras_val.png',dpi=150, bbox_inches='tight')
  plt.close()

    
  ker_loss = np.zeros((1,8))
  for i in range(8):
    ker_loss[0,i] = hist.history['model_2_'+str(i+1)+'_loss'][0]
  if keras_loss is None:
    keras_loss = np.copy(ker_loss)
  else:
    keras_loss = np.concatenate((keras_loss,ker_loss),axis=0)      
  plt.figure()
  plt.plot(keras_loss)
  plt.xlabel('epochs')
  plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
  plt.savefig(output_path+prefix+'_keras_loss.png',dpi=150, bbox_inches='tight')
  plt.close()
 

#%%

  #Checking if the model is reversible
  x = Input(shape=init_shape)	
  fx= feature_model(x)
  mx= mapping_x2y(fx)
  my= mapping_y2x(mx)
  tmpmodel = Model(inputs=x, outputs=[fx,my])
  tmpmodel.compile(optimizer=Adam(lr=learning_rate), 
                  loss=loss)

  [a,b] = tmpmodel.predict(x=t2, batch_size = batch_size)
  res = np.mean((a - b)**2)
  print('Error on reversible mapping in feature space: '+str(res))
  inverr.append(res)
  plt.figure()
  plt.plot(inverr)
  plt.legend(['$\| f(x) - m^{-1} \circ m \circ f(x) \|^2$'])
  plt.savefig(output_path+prefix+'_inverr.png',dpi=150, bbox_inches='tight')
  plt.close()    

#Save all data
joblib.dump((feature_model,mapping_x2y,mapping_y2x,recon_model),output_path+prefix+'_models.pkl', compress=True)
joblib.dump((keras_loss,keras_val,inverr,losses,psnrs),output_path+prefix+'_history.pkl', compress=True)

#%%  

# #Apply on a test image
# T1image = nibabel.load(output_path+'IXI661-HH-2788-T1_N4.nii.gz')
# T2image = nibabel.load(output_path+'IXI661-HH-2788-T2_N4.nii.gz')
# PDimage = nibabel.load(output_path+'IXI661-HH-2788-PD_N4.nii.gz')
# maskarray = nibabel.load(output_path+'IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)
