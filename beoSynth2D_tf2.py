#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:18:06 2019

@author: rousseau
"""

#%%
import joblib
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from model_zoo import build_feature_model_2d, build_recon_model_2d, build_block_model_2d, conv_bn_relu_nd
from model_zoo import build_encoder_2d, build_decoder_2d
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition, build_model
from model_zoo import build_exp_model, fm_model, build_reversible_forward_model_2d, build_reversible_backward_model_2d, build_4_model
from tensorflow.keras.optimizers import Adam, SGD
#from tensorflow.keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
from dataset import get_ixi_2dpatches, get_hcp_2dpatches, get_hcp_4darrays
#from sklearn.model_selection import train_test_split
import subprocess
import socket
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler

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
n_filters = 128
n_layers = 10          #number of layers for the numerical integration scheme (ResNet)
n_layers_residual = 1  #number of layers for the parametrization of the velocity fields
block_type = 'bottleneck' #what choice to do ?
loss = 'mse' 
batch_size = 8  
epochs = 100
use_optim = 0
kernel_size = 5
shared_blocks = 0  
ki = 'glorot_normal' #kernel initializer
ar = 0#1e-6                #activity regularization (L2)
#Get automatically the number of GPUs : https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
n_gpu = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
optim = 'adam'
lr_decay= 'none'

learning_rate = 0.0001   
if optim == 'sgd':
  learning_rate = 0.001 #changer le lr ? et l'adapter en fonction des epochs?
 
lw1 = 1 #loss weight direct mapping
lw2 = 1 #loss weight inverse mapping
lw3 = 1 #loss weight identity mapping on x 
lw4 = 1 #loss weight identity mapping on y
lw5 = 1 #loss weight cycle for x
lw6 = 1 #loss weight cycle for y
lw7 = 1 #loss weight autoencoder for x
lw8 = 1 #loss weight autoencoder for y

lws = [lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8]
   
inverse_consistency = 0
cycle_consistency = 0
identity_consistency = 0
reversible = 1 
backward_order = 3
encdec = 1
n_levels = 1    
zero_output = False #to be modified if necessary

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
else:
  prefix += '_order'+str(backward_order)  
if shared_blocks == 1:
  prefix += '_shared'  
prefix+= '_'+ki
prefix+= '_'+loss
prefix+= '_ar'+str(ar)  
prefix+= '_e'+str(epochs)+'_ps'+str(patch_size)+'_np'+str(x_train.shape[0])
prefix+= '_bs'+str(batch_size)
prefix+= '_lr'+str(learning_rate)
prefix+= '_nf'+str(n_filters)
prefix+= '_nl'+str(n_layers)
prefix+= '_nlr'+str(n_layers_residual)
prefix+= '_'+block_type
prefix+= '_lw'+str(lw1)+'_'+str(lw2)+'_'+str(lw3)+'_'+str(lw4)+'_'+str(lw5)+'_'+str(lw6)+'_'+str(lw7)+'_'+str(lw8)
prefix+= '_encdec'+str(encdec)
prefix+= '_nlevels'+str(n_levels)
prefix+= '_'+optim
prefix+= '_'+lr_decay
prefix+= '_'

print(prefix)

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
strategy = tf.distribute.MirroredStrategy()
batch_size = batch_size * n_gpu
print('Adapted batch size wrt number of GPUs : '+str(batch_size))
print('Number of GPUs used:'+str(strategy.num_replicas_in_sync))

with strategy.scope():
  
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
        block_f_x2y.append(build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar))
        block_g_x2y.append(build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar))
    else:
      bf = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar)
      bg = build_block_model_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar)
      for l in range(n_layers):
        block_f_x2y.append(bf)
        block_g_x2y.append(bg)
  
    mapping_x2y = build_reversible_forward_model_2d(init_shape=feature_shape, block_f=block_f_x2y, block_g = block_g_x2y, n_layers=n_layers)
    mapping_y2x = build_reversible_backward_model_2d(init_shape=feature_shape, block_f=block_f_x2y, block_g = block_g_x2y, n_layers=n_layers)
  
  else:
    block_x2y = []
    if shared_blocks == 0:
      for l in range(n_layers):
        block_x2y.append( build_block_model_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar) )
    else:
      b = build_block_model_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar)
      for l in range(n_layers):
        block_x2y.append(b)
  
    mapping_x2y = build_forward_model(init_shape=feature_shape, block_model=block_x2y, n_layers=n_layers)
    mapping_y2x = build_backward_model(init_shape=feature_shape, block_model=block_x2y, n_layers=n_layers, order=backward_order)
    
  
  model_all_gpu = build_4_model(init_shape, feature_model, mapping_x2y, mapping_y2x, recon_model)
  
  #Select the model to optimze wrt.
  # if n_gpu > 1:
  #   #tf.compat.v1.disable_eager_execution()
  #   model_all_gpu = multi_gpu_model(model_all, gpus=n_gpu)
  #   batch_size = batch_size * n_gpu
  # else:
  #   model_all_gpu = model_all
  # model_all_gpu.compile(optimizer=Adam(lr=learning_rate), loss=loss, loss_weights=lws)
  
  #transition to tf2 : 
  #model_all_gpu = model_all
    
  # if lr_decay == 'cosine':
  #   learning_rate = tf.keras.experimental.CosineDecay(learning_rate,100)
  if optim == 'adam':
    optimizer = Adam(lr=learning_rate)
  if optim == 'sgd':
    optimizer = SGD(lr=learning_rate)
    
  model_all_gpu.compile(optimizer=optimizer, loss=loss, loss_weights=lws) 
  print('Initial learning rate:'+str(K.eval(model_all_gpu.optimizer.lr)))
  
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

keras_val = None
keras_loss = None
losses = None
psnrs  = None
inverr = []
callbacks = []
# if lr_decay == 'cosine':
#   lr_decayed = tf.keras.experimental.CosineDecay(initial_learning_rate = learning_rate, decay_steps = 1000)
#   callbacks = [lr_decayed]
if lr_decay == 'decay':
  def lr_decayed(epoch):
    return learning_rate * (0.9 ** epoch)
  
  #lr_decayed = callbacks.LearningRateScheduler(schedule=lambda epoch: learning_rate * (0.9 ** epoch))
  callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_decayed))
    
dyn_t1 = np.max(y_train) - np.min(y_train)
dyn_t2 = np.max(x_train) - np.min(x_train)
print('T1 dynamic :'+str(dyn_t1))
print('T2 dynamic :'+str(dyn_t2))

n = 10
step = 4000
t2 = x_test[0:n*step:step,:,:,:]
t1 = y_test[0:n*step:step,:,:,:]

#%%
#Simple direct to find the best parameters
# with strategy.scope():
#   ix = Input(shape=init_shape)	 
#   fx= feature_model(ix)
#   mx= mapping_x2y(fx)
#   rx= recon_model(mx)
#   modelgx = Model(inputs=ix, outputs=rx)

# #Test full CNN for best performance
# from tensorflow.keras.layers import Conv2D

# with strategy.scope():
#   ix = Input(shape=init_shape)
#   x = conv_bn_relu_nd(ix, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   x = conv_bn_relu_nd(x, dimension=2, n_filters=n_filters)
#   output=(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)
#   modelgx =  Model(ix,output)

#   modelgx.compile(optimizer=Adam(lr=learning_rate), loss=loss)

#   print('Number of parameters of gx model : '+str(modelgx.count_params()))

#   hist = modelgx.fit(x=x_train, 
#                             y=y_train, 
#                             batch_size=batch_size, 
#                             epochs=1,  
#                             verbose=1, 
#                             shuffle=True,
#                             validation_data=(x_test,y_test))
# print(hist.history)
# plt.figure()
# plt.plot(10*np.log10( dyn_t1 / hist.history['val_loss'] ))
# plt.xlabel('epochs')
# plt.legend(['$g(x)$'])
# plt.savefig(output_path+prefix+'_gxcnn_psnr.png',dpi=150, bbox_inches='tight')
# plt.close() 


# # #%%
#import sys
#sys.exit()

#%%

class SaveFigureCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.train_loss = None
    self.val_loss = None
    self.train_psnr = None
    self.val_psnr = None
    self.inverr = []
    
  def on_epoch_end(self,epoch,logs):
    
    klist = list(logs.keys())
    
    print('\n Saving figures...')
    ker_loss = np.zeros((1,8))
    for i in range(8):
      ker_loss[0,i] = logs[klist[1+i]]
    if self.train_loss is None:
      self.train_loss = np.copy(ker_loss)
    else:
      self.train_loss = np.concatenate((self.train_loss,ker_loss),axis=0)      
    plt.figure()
    plt.plot(self.train_loss)
    plt.xlabel('epochs')
    plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
    plt.savefig(output_path+prefix+'_keras_loss.png',dpi=150, bbox_inches='tight')
    plt.close()    
    
    val_loss = np.zeros((1,8))
    for i in range(8):
      val_loss[0,i] = logs[klist[10+i]]
    if self.val_loss is None:
      self.val_loss = np.copy(val_loss)
    else:
      self.val_loss = np.concatenate((self.val_loss,val_loss),axis=0)  
    plt.figure()
    plt.plot(self.val_loss)
    plt.xlabel('epochs')
    plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
    plt.savefig(output_path+prefix+'_keras_val.png',dpi=150, bbox_inches='tight')
    plt.close()

    psnr = np.zeros((1,8))
    psnr[0,0]= 10*np.log10( dyn_t1 / logs[klist[1]] )
    psnr[0,1]= 10*np.log10( dyn_t2 / logs[klist[2]] )  
    psnr[0,2]= 10*np.log10( dyn_t2 / logs[klist[3]] )  
    psnr[0,3]= 10*np.log10( dyn_t1 / logs[klist[4]] )  
    psnr[0,4]= 10*np.log10( dyn_t2 / logs[klist[5]] ) 
    psnr[0,5]= 10*np.log10( dyn_t1 / logs[klist[6]] )  
    psnr[0,6]= 10*np.log10( dyn_t2 / logs[klist[7]] )  
    psnr[0,7]= 10*np.log10( dyn_t1 / logs[klist[8]] )
    if self.train_psnr is None:
      self.train_psnr = np.copy(psnr)
    else:
      self.train_psnr = np.concatenate((self.train_psnr,psnr),axis=0)      
    plt.figure()
    plt.plot(self.train_psnr)
    plt.xlabel('epochs')
    plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
    plt.savefig(output_path+prefix+'_train_psnr.png',dpi=150, bbox_inches='tight')
    plt.close()    

    psnr = np.zeros((1,8))
    psnr[0,0]= 10*np.log10( dyn_t1 / logs[klist[10]] )
    psnr[0,1]= 10*np.log10( dyn_t2 / logs[klist[11]] )  
    psnr[0,2]= 10*np.log10( dyn_t2 / logs[klist[12]] )  
    psnr[0,3]= 10*np.log10( dyn_t1 / logs[klist[13]] )  
    psnr[0,4]= 10*np.log10( dyn_t2 / logs[klist[14]] ) 
    psnr[0,5]= 10*np.log10( dyn_t1 / logs[klist[15]] )  
    psnr[0,6]= 10*np.log10( dyn_t2 / logs[klist[16]] )  
    psnr[0,7]= 10*np.log10( dyn_t1 / logs[klist[17]] )
    if self.val_psnr is None:
      self.val_psnr = np.copy(psnr)
    else:
      self.val_psnr = np.concatenate((self.val_psnr,psnr),axis=0)      
    plt.figure()
    plt.plot(self.val_psnr)
    plt.xlabel('epochs')
    plt.legend(['$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'])
    plt.savefig(output_path+prefix+'_val_psnr.png',dpi=150, bbox_inches='tight')
    plt.close()    
    
    print('Checking if the model is reversible')
    ix = Input(shape=init_shape)	
    fx= feature_model(ix)
    mx= mapping_x2y(fx)
    my= mapping_y2x(mx)
    tmpmodel = Model(inputs=ix, outputs=[fx,my])
    tmpmodel.compile(optimizer=Adam(lr=learning_rate), 
                    loss=loss)
  
    [a,b] = tmpmodel.predict(x=t2, batch_size = batch_size)
    res = np.mean((a - b)**2)
    print('Error on reversible mapping in feature space: '+str(res))
    self.inverr.append(res)
    plt.figure()
    plt.plot(self.inverr)
    plt.legend(['$\| f(x) - m^{-1} \circ m \circ f(x) \|^2$'])
    plt.savefig(output_path+prefix+'_inverr.png',dpi=150, bbox_inches='tight')
    plt.close()   

    print('Saving predicted patches') 
    [a,b,c,d,e,f,g,h] = model_all_gpu.predict(x=[t2,t1], batch_size = batch_size)
    show_patches(patch_list=[t2,t1,a,b,c,d,e,f,g,h],filename=output_path+prefix+'_current'+str(epoch)+'fig_patches.png')



callbacks.append(SaveFigureCallback())

x = [x_train,y_train]
y = [y_train,x_train,x_train,y_train,x_train,y_train,x_train,y_train]
x_val = [x_test,y_test]
y_val = [y_test,x_test,x_test,y_test,x_test,y_test,x_test,y_test]

print('Learning model all gpu')
hist = model_all_gpu.fit(x=x, 
                          y=y, 
                          batch_size=batch_size, 
                          epochs=epochs, 
                          verbose=2, 
                          shuffle=True,
                          validation_data=(x_val,y_val),
                          callbacks=callbacks)


#Save all data
#Pickle -> bug ?!! (we save hist.history instead of hist)
joblib.dump(hist.history,output_path+prefix+'_kerashistory.pkl', compress=True)
joblib.dump((keras_loss,keras_val,inverr,losses,psnrs),output_path+prefix+'_myhistory.pkl', compress=True)
model_all_gpu.save(output_path+prefix+'_model.h5')
feature_model.save(output_path+prefix+'_feature_model.h5')
mapping_x2y.save(output_path+prefix+'_mapping_x2y.h5')
mapping_y2x.save(output_path+prefix+'_mapping_y2x.h5')
recon_model.save(output_path+prefix+'_recon_model.h5')



#joblib.dump((feature_model,mapping_x2y,mapping_y2x,recon_model),output_path+prefix+'_models.pkl', compress=True)



#%%  

# #Apply on a test image
# T1image = nibabel.load(output_path+'IXI661-HH-2788-T1_N4.nii.gz')
# T2image = nibabel.load(output_path+'IXI661-HH-2788-T2_N4.nii.gz')
# PDimage = nibabel.load(output_path+'IXI661-HH-2788-PD_N4.nii.gz')
# maskarray = nibabel.load(output_path+'IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)
