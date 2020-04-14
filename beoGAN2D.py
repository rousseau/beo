#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
from os.path import expanduser
home = expanduser("~")

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D, Activation, Reshape
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Flatten, Dense, Lambda, concatenate, Multiply, Add, Subtract
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from model_zoo import build_4_model, build_encoder_2d, build_decoder_2d, build_block_model_2d, build_reversible_forward_model_2d, build_reversible_backward_model_2d, UNet

import types 
import socket
import sys 
import datetime

def wasserstein_loss(y_true, y_pred):
  return K.mean(y_true * y_pred)

#%%

dataset = 'HCP' #HCP or IXI or Bazin  
patch_size = 64 

if dataset == 'IXI':  
  X_train = joblib.load(home+'/Exp/IXI_T1_2D_'+str(patch_size)+'_training.pkl')
  Y_train = joblib.load(home+'/Exp/IXI_T2_2D_'+str(patch_size)+'_training.pkl')
if dataset == 'HCP':  
  X_train = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_training.pkl')
  Y_train = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_training.pkl')
  X_test = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_testing.pkl')
  Y_test = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_testing.pkl')
if dataset == 'Bazin':
  X_train = joblib.load(home+'/Exp/BlockFace_'+str(patch_size)+'_training.pkl')
  Y_train = joblib.load(home+'/Exp/Thio_'+str(patch_size)+'_training.pkl')

#%%

multigpu = 0
mem_limit = 0
use_fp16 = 0 

#Limited memory growth -> allows to check the memory required in each GPU
if mem_limit == 1:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

if use_fp16 == 1:  
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)
  print('Compute dtype: %s' % policy.compute_dtype) 
  print('Variable dtype: %s' % policy.variable_dtype)

#%%
batch_size = 16  
epochs = 8       #n_epochs per learning steps
n_filters = 32
n_channelsX = 1
n_channelsY = 1 

#Mapping parameters
n_layers_list = [8]#[1,2,4,8,16,32,64]
n_layers = np.max(n_layers_list)           #number of layers for the numerical integration scheme (ResNet)
n_layers_residual = 2  #number of layers for the parametrization of the velocity fields

reversible_mapping = 0 
shared_blocks = 0
backward_order = 1     #Behrmann et al. ICML 2019 used 100
scaling_weights = 0

#Discriminator parameters
n_levels_discriminator = 4
n_filters_discriminator = 128
patch_size_discriminator = int(patch_size / 2**n_levels_discriminator)
loss_discriminator = 'mse'#wasserstein_loss#'binary_crossentropy' #'mse' or 'binary_crossentropy'
discriminator_regularizer = 0.01
clip_value = 0.01
clipping = False
n_critic_steps = 1


ki = 'glorot_normal' #kernel initializer
ar = 1e-5  #1e-6                #activity regularization (L2)
kr = 0                     #kernel regularization (L1)
optimizer = Adam(0.0001)#RMSprop(lr=0.00005)#Adam(0.0002, 0.5)

lambda_adv = 0.1
lambda_direct = 1
lambda_cycle = 1
lambda_id = 1# * lambda_cycle
lambda_ae = 1  
save_interval=10000 
pre_generator_training = False
pre_discriminator_training = False

lw1 = lambda_direct #loss weight direct mapping
lw2 = lambda_direct #loss weight inverse mapping
lw3 = lambda_id #loss weight identity mapping on x 
lw4 = lambda_id #loss weight identity mapping on y
lw5 = lambda_cycle #loss weight cycle for x
lw6 = lambda_cycle #loss weight cycle for y
lw7 = lambda_ae #loss weight autoencoder for x
lw8 = lambda_ae #loss weight autoencoder for y

lws = [lw1,lw2,lw3,lw4,lw5,lw6,lw7,lw8]

#Create a suffix to record results
p = types.SimpleNamespace() #All parameters in p
#p.n_critic_steps = n_critic_steps
#p.n_filters_discriminator = n_filters_discriminator
if use_fp16 == 1:
  p.use_fp16 = use_fp16
p.epochs = epochs
p.batch_size = batch_size
p.n_filters = n_filters
p.n_layers = n_layers
if len(n_layers_list)>1:
  p.prog_learning = 1
p.reversible_mapping = reversible_mapping
if reversible_mapping == 0:
  p.backward_order = backward_order
p.shared_blocks = shared_blocks
p.scaling_weights = scaling_weights
p.ar = ar
#p.lambda_direct = lambda_direct
#p.lambda_cycle = lambda_cycle
#p.lambda_id = lambda_id
#p.lambda_ae = lambda_ae

if 'Precision' in socket.gethostname():
  prefix = 'dell_'
else:
  prefix = 'aorus_'
  
dic = vars(p)
for k in dic.keys():
  prefix += k+'_'+str(dic[k])+'_'
print(prefix)  

output_path = home+'/Sync/Experiments/'+dataset+'/'


#%%


if K.image_data_format() == 'channels_first':
  init_shape = (n_channelsX, None, None)
  feature_shape = (n_filters, None, None)
  half_feature_shape = ((int)(n_filters/2), None, None)
  channel_axis = 1
else:
  init_shape = (None, None,n_channelsX)
  #init_shape = (patch_size, patch_size,n_channelsX) # pourquoi?
  feature_shape = (None, None, n_filters)
  half_feature_shape = (None, None, (int)(n_filters/2))
  channel_axis = -1  

print(K.image_data_format())

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
if multigpu == 1:
  strategy = tf.distribute.MirroredStrategy()

  import subprocess
  n_gpu = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
  batch_size = batch_size * n_gpu
  print('Adapted batch size wrt number of GPUs : '+str(batch_size))
  print('Number of GPUs used:'+str(strategy.num_replicas_in_sync))

#%%
def conv_relu_block_2d(input,n_filters):
  n = Conv2D(n_filters, 3, activation='relu', padding='same')(input)
  n = Conv2D(n_filters, 3, activation='relu', padding='same')(n)
  return n

def build_feature_model(init_shape=init_shape, n_filters=n_filters):
  inputs = Input(shape=init_shape)  
  features = inputs

  #Multiscales

  #features = conv_relu_block_2d(features, n_filters)    
  #Downsampling used to decrease GPU memory
  features = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='glorot_uniform',
                  use_bias=True,
                  strides=2,
                  activation='relu')(features)
  features = conv_relu_block_2d(features, n_filters)    

  features = Activation('tanh')(features)
  
  # if use_fp16 == 1:
  #   features = tf.keras.layers.Activation('linear', dtype='float32')(features)
  
  model = Model(inputs=inputs, outputs=features)
  
  #return sparsity.prune_low_magnitude(model, **pruning_params)
  return model

def build_recon_model(init_shape, n_filters=32):  
  inputs = Input(shape=init_shape)
  recon = inputs

  #recon = conv_relu_block_2d(recon, n_filters)    
  recon = Conv2DTranspose(filters=n_filters, kernel_size=3, padding='same', kernel_initializer='glorot_uniform',
                  use_bias=True,
                  strides=2,
                  activation='relu')(recon)
  recon = conv_relu_block_2d(recon, n_filters)    
  recon = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer='glorot_uniform',
                    use_bias=True,
                    strides=1,
                    activation='linear')(recon)
  
  if use_fp16 == 1:
    recon = tf.keras.layers.Activation('linear', dtype='float32')(recon)
  
  model = Model(inputs=inputs, outputs=recon)
  return model

def build_block_mapping_2d(init_shape,
                         n_filters=32, 
                         n_layers=2, 
                         kernel_initializer = 'glorot_uniform',
                         activity_regularizer = 0,
                         kernel_regularizer = 0):
  
  inputs = Input(shape=init_shape)
  x = inputs

  for i in range(n_layers-1):
    x = Conv2D((int)(n_filters), (3, 3), padding='same', kernel_initializer=kernel_initializer,
                    use_bias=False,
                    strides=1,
                    activity_regularizer = tf.keras.regularizers.l2(activity_regularizer),
                    kernel_regularizer = tf.keras.regularizers.l1(kernel_regularizer))(x)
    x = Activation('relu')(x)

  x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                  use_bias=False,
                  strides=1,
                  activity_regularizer = tf.keras.regularizers.l2(activity_regularizer),
                  kernel_regularizer = tf.keras.regularizers.l1(kernel_regularizer))(x)
  
  x = Activation('tanh')(x) #required for approximation of the inverse
    
  outputs = x  
  model = Model(inputs=inputs, outputs=outputs)
  return model

def build_forward_block_mapping_2d(init_shape, block):
  # mapping type : 1 (reversible architecture), 0 (residual networks)
  # block : 2 blocks (reversible) or 1 block (residual networks)
  
  if len(block) == 1:
    reversible_mapping = 0
  else:
    reversible_mapping = 1
    
  inputs = Input(shape=init_shape)
  
  if reversible_mapping == 0:
    x = block[0](inputs)
    outputs = Add()([inputs,x])
    
  else:
    #Split channels
    x = inputs
    if K.image_data_format() == 'channels_last':
      x1 = Lambda(lambda x: x[:,:,:,:(int)(K.int_shape(x)[-1]/2)])(x)
      x2 = Lambda(lambda x: x[:,:,:,(int)(K.int_shape(x)[-1]/2):])(x)
      channel_axis = -1
    else:
      x1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:])(x)
      x2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:])(x)
      channel_axis = 1
      
    xx = block[0](x2)
    y1= Add()([x1,xx])
    
    xx = block[1](y1)
    y2= Add()([x2,xx])

    x1 = y1
    x2 = y2    
  
    outputs = concatenate([x1,x2], axis=channel_axis)    
    
  model = Model(inputs=inputs, outputs=outputs)
  return model

def build_backward_block_mapping_2d(init_shape, block, order = 1):
  # mapping type : 1 (reversible architecture), 0 (residual networks)
  # block : 2 blocks (reversible) or 1 block (residual networks)
  # order : approximation order for backward block for residual network

  if len(block) == 1:
    reversible_mapping = 0
  else:
    reversible_mapping = 1
    
  inputs = Input(shape=init_shape)
  
  if reversible_mapping == 0:
    x = inputs
    #Third order approximation of v, the inverse of u (assumed to be linear !!!).
    #v = -u + u² -u³ ...
    
    #move to fixed point iterations
    for i in range(order):

      u = block[0](x)
      x = Subtract()([inputs,u])

    # if order>1: 
    #   u2 = Multiply()([u,u])
    #   x = Add()([x,u2])
    # if order>2:
    #   u3 = Multiply()([u2,u])    
    #   x = Subtract()([x,u3])   
       
    outputs = x
    
  else:
    #Split channels
    y = inputs
    if K.image_data_format() == 'channels_last':
      y1 = Lambda(lambda x: x[:,:,:,:(int)(K.int_shape(x)[-1]/2)])(y)
      y2 = Lambda(lambda x: x[:,:,:,(int)(K.int_shape(x)[-1]/2):])(y)
      channel_axis = -1
    else:
      y1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:])(y)
      y2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:])(y)
      channel_axis = 1
      
    yy = block[1](y1)
    x2= Subtract()([y2,yy])
    
    yy = block[0](x2)
    x1= Subtract()([y1,yy])

    y1 = x1
    y2 = x2    
  
    outputs = concatenate([y1,y2], axis=channel_axis)    
    
  model = Model(inputs=inputs, outputs=outputs)
  return model

def build_mapping(init_shape, blocks):
  inputs = Input(shape=init_shape)
  x = inputs

  for i in range(len(blocks)):    
    x = blocks[i](x)

  outputs = x
  
  model = Model(inputs=inputs, outputs=outputs)
  return model

def build_direct_generator(init_shape, feature_model, mapping, reconstruction_model):
  ix = Input(shape=init_shape)	
  fx = feature_model(ix)   
  mx2y = mapping(fx)
  rx2y = reconstruction_model(mx2y)

  model = Model(inputs=[ix], 
                outputs=[rx2y])  
  return model

def build_cycle_generator(init_shape, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model):
  ix = Input(shape=init_shape)	 
  fx = feature_model(ix)   
  mx2y = mapping_x_to_y(fx)
  rx2y = reconstruction_model(mx2y)
  frx2y = feature_model(rx2y)
  mx2y2x = mapping_y_to_x(frx2y)
  rx2y2x = reconstruction_model(mx2y2x)  

  model = Model(inputs=[ix], 
                outputs=[rx2y2x])  
  return model

def build_ae_generator(init_shape, feature_model, reconstruction_model):  
  ix = Input(shape=init_shape)	 
  fx = feature_model(ix)   
  idx2x = reconstruction_model(fx)  

  model = Model(inputs=[ix], 
                outputs=[idx2x])  
  return model

def zero_loss_generator(init_shape, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model):
  ix = Input(shape=init_shape)	
  iy = Input(shape=init_shape)	

# lw1 = lambda_direct #loss weight direct mapping
# lw2 = lambda_direct #loss weight inverse mapping
# lw3 = lambda_id #loss weight identity mapping on x 
# lw4 = lambda_id #loss weight identity mapping on y
# lw5 = lambda_cycle #loss weight cycle for x
# lw6 = lambda_cycle #loss weight cycle for y
# lw7 = lambda_ae #loss weight autoencoder for x
# lw8 = lambda_ae #loss weight autoencoder for y
  errors = []
  if lw1 > 0 : 
    model1 = build_direct_generator(init_shape, feature_model, mapping_x_to_y, reconstruction_model)
    rx2y = model1(ix)
    err = Subtract()([rx2y,iy])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)
    
  if lw2 > 0 : 
    model2 = build_direct_generator(init_shape, feature_model, mapping_y_to_x, reconstruction_model)
    ry2x = model2(iy)
    err = Subtract()([ry2x,ix])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  if lw3 > 0 :
    model3 = build_direct_generator(init_shape, feature_model, mapping_y_to_x, reconstruction_model)
    rx2x = model3(ix)
    err = Subtract()([rx2x,ix])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  if lw4 > 0 :
    model4 = build_direct_generator(init_shape, feature_model, mapping_x_to_y, reconstruction_model)
    ry2y = model4(iy)
    err = Subtract()([ry2y,iy])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  if lw5 > 0 :
    model5 = build_cycle_generator(init_shape, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model)
    rx2y2x = model5(ix)
    err = Subtract()([rx2y2x,ix])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  if lw6 > 0 :
    model6 = build_cycle_generator(init_shape, feature_model, mapping_y_to_x, mapping_x_to_y, reconstruction_model)
    ry2x2y = model6(iy)
    err = Subtract()([ry2x2y,iy])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  if lw7 > 0 :
    model7 = build_ae_generator(init_shape, feature_model, reconstruction_model)
    idx2x = model7(ix)
    err = Subtract()([idx2x,ix])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  if lw8 > 0 :
    model8 = build_ae_generator(init_shape, feature_model, reconstruction_model)
    idy2y = model8(iy)
    err = Subtract()([idy2y,iy])
    err = Lambda(lambda x:K.pow(x,2))(err)
    err = GlobalAveragePooling2D()(err)
    err = Reshape((1,))(err)
    errors.append(err)

  errsum = Add()(errors)
  if use_fp16 == 1:
    errsum = tf.keras.layers.Activation('linear', dtype='float32')(errsum)
    
  return Model(inputs=[ix,iy], outputs=errsum)  
  
def build_generator(init_shape, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model):
  ix = Input(shape=init_shape)	
  iy = Input(shape=init_shape)	
  
  fx = feature_model(ix)   
  fy = feature_model(iy)
  
  #Forward     
  mx2y = mapping_x_to_y(fx)
  rx2y = reconstruction_model(mx2y)
  #Backward
  my2x = mapping_y_to_x(fy)
  ry2x = reconstruction_model(my2x)

  #Identity mapping constraints
  my2y = mapping_x_to_y(fy)
  ry2y = reconstruction_model(my2y)

  mx2x = mapping_y_to_x(fx)
  rx2x = reconstruction_model(mx2x) 

  #Cycle consistency
  frx2y = feature_model(rx2y)
  mx2y2x = mapping_y_to_x(frx2y)
  rx2y2x = reconstruction_model(mx2y2x)  

  fry2x = feature_model(ry2x)
  my2x2y = mapping_x_to_y(fry2x)
  ry2x2y = reconstruction_model(my2x2y)  

  #Autoencoder constraints
  idy2y = reconstruction_model(fy)
  idx2x = reconstruction_model(fx)  

  model = Model(inputs=[ix,iy], 
                outputs=[rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y])
  
  return model

def build_intermediate_model(init_shape, feature_model, mapping):
  ix = Input(shape=init_shape)	
  fx = feature_model(ix)   
  mx2y = mapping(fx)
  if use_fp16 == 1:
    mx2y = tf.keras.layers.Activation('linear', dtype='float32')(mx2y)

  model = Model(inputs=[ix], 
                outputs=[mx2y])  
  return model
  
#%%



def build_mapping_blocks():
  forward_blocks = []
  backward_blocks= []
  block_x2y_list = []

  if shared_blocks == 1:
    if reversible_mapping == 1:
      block_f_x2y = build_block_mapping_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar, kernel_regularizer=kr)
      block_g_x2y = build_block_mapping_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar, kernel_regularizer=kr)
      block_x2y = [block_f_x2y,block_g_x2y]
      
    else:
      block_x2y = [build_block_mapping_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar, kernel_regularizer=kr)]
      
    forward_block = build_forward_block_mapping_2d(init_shape=feature_shape, block=block_x2y)
    backward_block = build_backward_block_mapping_2d(init_shape=feature_shape, block=block_x2y, order = backward_order)    
    block_x2y_list.append(block_x2y)
      
    for l in range(n_layers):
      forward_blocks.append(forward_block)
      backward_blocks.append(backward_block)

  else:
    for l in range(n_layers):
      if reversible_mapping == 1:
        block_f_x2y = build_block_mapping_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar, kernel_regularizer=kr)
        block_g_x2y = build_block_mapping_2d(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar, kernel_regularizer=kr)
        block_x2y = [block_f_x2y,block_g_x2y]
        
      else:
        block_x2y = [build_block_mapping_2d(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual, kernel_initializer=ki, activity_regularizer=ar, kernel_regularizer=kr)]
        
      forward_block = build_forward_block_mapping_2d(init_shape=feature_shape, block=block_x2y)
      backward_block = build_backward_block_mapping_2d(init_shape=feature_shape, block=block_x2y, order = backward_order)    
      
      forward_blocks.append(forward_block)
      backward_blocks.append(backward_block)
      block_x2y_list.append(block_x2y)
  return (forward_blocks, backward_blocks, block_x2y_list)  

def build_all_models():
  feature_model = build_feature_model(init_shape=init_shape, n_filters=n_filters)  
  recon_model = build_recon_model(init_shape=feature_shape, n_filters=n_filters) 
 
  forward_blocks, backward_blocks, block_x2y_list = build_mapping_blocks()

  return (feature_model, recon_model, forward_blocks, backward_blocks, block_x2y_list)  

        
  # mapping_x2y = build_mapping(init_shape=feature_shape, blocks = forward_blocks)
  # mapping_y2x = build_mapping(init_shape=feature_shape, blocks = backward_blocks)
      
  # #Build intermediate models 
  # for l in range(n_layers):
  #   intermediate_models.append( build_intermediate_model(init_shape, feature_model, build_mapping(init_shape=feature_shape, blocks = forward_blocks[:(l+1)]) ) )

  # paired_generator = build_generator(init_shape, feature_model, mapping_x2y, mapping_y2x, recon_model)
  # paired_generator.compile(optimizer=optimizer, loss='mse', loss_weights=lws) 
  # paired_generator.summary()
  
  # zero_loss_model = zero_loss_generator(init_shape, feature_model, mapping_x2y, mapping_y2x, recon_model)
 
  # return (feature_model, recon_model, mapping_x2y, mapping_y2x, paired_generator,zero_loss_model)  

# if multigpu == 1:  
#   with strategy.scope():
#     models = build_all_models()
# else:
#   models = build_all_models()    

# (feature_model, recon_model, mapping_x2y, mapping_y2x, paired_generator,zero_loss_model) = models  

#%%
intermediate_models = []

if multigpu == 1:  
  with strategy.scope():
    models = build_all_models()
else:
  models = build_all_models()    

(feature_model, recon_model, forward_blocks, backward_blocks, block_x2y_list) = models  

  
#%%
import scipy.ndimage

def compute_lipschitz_conv_weights(weights):  
  L = 0
  n = 20
  dim_features = weights.shape[2]
  u = np.random.rand(n,n,dim_features)
  v = np.random.rand(n,n,dim_features)
  W = weights
  n_iter = 50
  for i in range (n_iter):
    WTu = scipy.ndimage.convolve(np.flip(np.flip(u,axis=0),axis=1), W)
    v_new = WTu / np.linalg.norm(WTu.flatten())
    v = v_new
  
    Wv = scipy.ndimage.convolve(v, W)
    u_new = Wv / np.linalg.norm(Wv.flatten())
    u = u_new
    
  Wv = scipy.ndimage.convolve(v, W)
  L = np.dot( np.transpose(u.flatten()), Wv.flatten())

  return L

def compute_lipschitz_conv_keras(weights, n_iter = 50):
  L = 0
  n_filters = weights[0].shape[2]
  n = 20
  u = np.random.rand(1,n,n,n_filters)
  v = np.random.rand(1,n,n,n_filters)

  f_in = Input(shape= (None, None, n_filters) )	
  f_out = Conv2D((int)(n_filters), (3, 3), padding='same', use_bias=False, strides=1)(f_in)
  conv_model = Model(inputs=f_in, outputs=f_out)
  conv_model.compile(optimizer=optimizer, loss='mse') 
  
  conv_model.layers[1].set_weights(weights)

  for i in range (n_iter):
    
    WTu = conv_model.predict(x=[ np.flip(np.flip(u,axis=1),axis=2) ])
    v_new = WTu / np.linalg.norm(WTu.flatten())
    v = v_new
  
    Wv = conv_model.predict(x=v)
    u_new = Wv / np.linalg.norm(Wv.flatten())
    u = u_new
    
  Wv = conv_model.predict(x=v)
  L = np.dot( np.transpose(u.flatten()), Wv.flatten())
  
  return L
 
       
#%%
callbacks = []

n_samples = X_test.shape[0]
#Take 10 examples for visualization
n = 10
step = (int)(n_samples / n)
t1_vis = X_test[0:n*step:step,:,:,:]
t2_vis = Y_test[0:n*step:step,:,:,:]

#Take more examples for callback computations
n = 100
step = (int)(n_samples / n)
t1 = X_test[0:n*step:step,:,:,:]
t2 = Y_test[0:n*step:step,:,:,:]

class SaveFigureCallback(tf.keras.callbacks.Callback):
    
  def on_epoch_end(self,epoch,logs):
    
    [a,b,c,d,e,f,g,h] = self.model.predict(x=[t1_vis,t2_vis], batch_size = batch_size)
    show_patches(patch_list=[t1_vis,t2_vis,a,b,c,d,e,f,g,h],
                 titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'],
                 filename=output_path+prefix+'_current'+str(epoch)+'fig_patches.png')

callbacks.append(SaveFigureCallback()) 

def save_figure(model, x, y, filename):
  [a,b,c,d,e,f,g,h] = model.predict(x=[x,y], batch_size = batch_size)
  show_patches(patch_list=[x,y,a,b,c,d,e,f,g,h],
               titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'],
               filename=filename)
  
#%%
print('Number of convolution blocks : '+str((len(block_x2y_list) * len(block_x2y_list[0] ))))
      
class LipschitzCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):    
    self.L = np.zeros( (len(block_x2y_list) * len(block_x2y_list[0] ), epochs) )
    
  def on_epoch_end(self,epoch,logs): 
    print('Computing Lipschitz constant')
    i = 0
    for blocks in block_x2y_list:
      for block in blocks:  
        cst = 1
        for layer in block.layers:
          if 'conv2d' in layer.name:
            tmp = compute_lipschitz_conv_keras(weights = layer.get_weights(), n_iter = 5)
            cst *= tmp
            #print('Lipschitz conv block '+str(i)+' at epoch '+str(epoch)+' : '+str(tmp))
        #print('Lipschitz combined conv block '+str(i)+' at epoch '+str(epoch)+' : '+str(cst))  
        self.L[i,epoch] = cst    
        i = i+1
    print('Max Lipschitz value : '+str( np.max(self.L[:,epoch]) ))

    plt.figure(figsize=(4,4))
    
    for i in range(self.L.shape[0]):
      plt.plot(self.L[i,0:epoch+1])
    plt.savefig(output_path+prefix+'_lipschitz.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    im = plt.imshow(self.L,cmap=plt.cm.gray, 
                 interpolation="nearest",
                 vmin=0,vmax=3)
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_lipschitz_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()
    

callbacks.append(LipschitzCallback())

def compute_lipschitz():
  L = []
  for blocks in block_x2y_list:
    for block in blocks:  
      cst = 1
      for layer in block.layers:
        if 'conv2d' in layer.name:
          tmp = compute_lipschitz_conv_keras(weights = layer.get_weights(), n_iter = 5)
          cst *= tmp
      L.append(cst)
  print('Lipschitz value : '+str(np.max(L)))
  return np.max(L)



#%%

class ReversibilityCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.inverr = []
    
  def on_epoch_end(self,epoch,logs): 
    print('Checking if the model is reversible')
    ix = Input(shape=init_shape)	
    fx= feature_model(ix)
    mx= mapping_x2y(fx)
    my= mapping_y2x(mx)
    if use_fp16 == 1:
      fx = tf.keras.layers.Activation('linear', dtype='float32')(fx)
      my = tf.keras.layers.Activation('linear', dtype='float32')(my)

    tmpmodel = Model(inputs=ix, outputs=[fx,my])
    tmpmodel.compile(optimizer=optimizer, 
                    loss='mse')
  
    [a,b] = tmpmodel.predict(x=t1, batch_size = batch_size)
    res = np.mean((a - b)**2)
    print('Error on reversible mapping in feature space: '+str(res))
    self.inverr.append(res)
    plt.figure()
    plt.plot(self.inverr)
    plt.legend(['$\| f(x) - m^{-1} \circ m \circ f(x) \|^2$'])
    plt.savefig(output_path+prefix+'_inverr.png',dpi=150, bbox_inches='tight')
    plt.close()   

callbacks.append(ReversibilityCallback())

def compute_reversibility_error():
  ix = Input(shape=init_shape)	
  fx= feature_model(ix)
  mx= mapping_x2y(fx)
  my= mapping_y2x(mx)
  if use_fp16 == 1:
    fx = tf.keras.layers.Activation('linear', dtype='float32')(fx)
    my = tf.keras.layers.Activation('linear', dtype='float32')(my)

  tmpmodel = Model(inputs=ix, outputs=[fx,my])
  tmpmodel.compile(optimizer=optimizer, 
                  loss='mse')

  [a,b] = tmpmodel.predict(x=t1, batch_size = batch_size)
  res = np.mean((a - b)**2)
  print('Reversibility error : '+str(res))
  return res  

#%%
#Compute the norm of the velocity field 
#(training or testing data?)
#Which norm?

class NormVelocityCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.norm_velocity = np.zeros((len(intermediate_models), epochs))
  
  def on_epoch_end(self,epoch,logs): 
    print('Computing norm of velocity field')
    for l in range(len(intermediate_models)):
      a = intermediate_models[l].predict(x=t1, batch_size = batch_size)
      self.norm_velocity[l,epoch] = np.max(np.linalg.norm(a.reshape((a.shape[0],-1)), ord=np.inf,axis=1))

    plt.figure(figsize=(4,4))
    
    for i in range(self.norm_velocity.shape[0]):
      plt.plot(self.norm_velocity[i,0:epoch+1])
    plt.savefig(output_path+prefix+'_norm_velocity.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))    
    im = plt.imshow(self.norm_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

callbacks.append(NormVelocityCallback())

def compute_norm_velocity():
  norm_velocity = []
  for l in range(len(intermediate_models)):
    a = intermediate_models[l].predict(x=t1, batch_size = batch_size)
    norm_velocity.append( np.max(np.linalg.norm(a.reshape((a.shape[0],-1)), ord=np.inf,axis=1)) )
  return norm_velocity
  

#%%
class GradientVelocityCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.norm_gradient_velocity = np.zeros((len(intermediate_models), epochs))  
  def on_epoch_end(self,epoch,logs): 
    print('Computing gradient of velocity field')
    for l in range(len(intermediate_models)):
      
      if use_fp16 == 1:
        inp = tf.Variable(np.random.normal(size= t1.shape), dtype=tf.float16)
      else:      
        inp = tf.Variable(np.random.normal(size= t1.shape), dtype=tf.float32)
      with tf.GradientTape() as tape:
        preds = intermediate_models[l](inp)
      grads = tape.gradient(preds,inp)      
      
      #In Tensorflow 1 : 
      #gradient = K.gradients(intermediate_models[l].output, intermediate_models[l].input)[2]
      #iterate = K.function(intermediate_models[l].input,[gradient])
      #grad = iterate([t1])
      a = grads.numpy()
      self.norm_gradient_velocity[l,epoch] = np.max(np.linalg.norm(a.reshape((a.shape[0],-1)), ord=np.inf,axis=1))

    plt.figure(figsize=(4,4))
    
    for i in range(self.norm_gradient_velocity.shape[0]):
      plt.plot(self.norm_gradient_velocity[i,0:epoch+1])
    plt.savefig(output_path+prefix+'_norm_gradient_velocity.png',dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(4,4))
    
    im = plt.imshow(self.norm_gradient_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_gradient_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

if multigpu == 0:
  callbacks.append(GradientVelocityCallback())

def compute_gradient_velocity():
  norm_gradient_velocity = []
  for l in range(len(intermediate_models)):
    
    if use_fp16 == 1:
      inp = tf.Variable(np.random.normal(size= t1.shape), dtype=tf.float16)
    else:      
      inp = tf.Variable(np.random.normal(size= t1.shape), dtype=tf.float32)
    with tf.GradientTape() as tape:
      preds = intermediate_models[l](inp)
    grads = tape.gradient(preds,inp)      
    
    a = grads.numpy()
    norm_gradient_velocity.append( np.max(np.linalg.norm(a.reshape((a.shape[0],-1)), ord=np.inf,axis=1)) )
  return norm_gradient_velocity  
  

#%%
x = [X_train,Y_train]
y = [Y_train,X_train,X_train,Y_train,X_train,Y_train,X_train,Y_train]

x_test = [X_test,Y_test]
y_test = [Y_test,X_test,X_test,Y_test,X_test,Y_test,X_test,Y_test]

#%%

print('Progressive Learning')
print(callbacks)
print('max n_layers : '+str(len(forward_blocks)))

hist = None

it = 0
n_epochs = 0
L = []
R = []
max_epochs = epochs * len(n_layers_list)
norm_velocity = np.zeros( (n_layers , max_epochs) )
norm_gradient_velocity = np.zeros( (n_layers, max_epochs) )  


for nl in range(len(n_layers_list)):
  print('*********************** NEW SCALE **********************************')
  n_blocks = n_layers_list[nl]
  print(str(len(forward_blocks[0:n_blocks]))+' mapping layers ')
  
  if scaling_weights == 1:
    if it > 0:
      scaling = n_layers_list[nl-1] * 1.0 / n_layers_list[nl]
      print('Doing weight scaling by '+str( scaling ))
      for blocks in block_x2y_list:
        for block in blocks:  
          for layer in block.layers:
            if 'conv2d' in layer.name:
              weights = layer.get_weights() #list of numpy array
              for w in weights:
                w *= scaling
              layer.set_weights(weights)      
  it += 1
  
  
  mapping_x2y = build_mapping(init_shape=feature_shape, blocks = forward_blocks[0:n_blocks])
  mapping_y2x = build_mapping(init_shape=feature_shape, blocks = backward_blocks[0:n_blocks])
  
  intermediate_models = []
  for l in range(n_blocks):
    intermediate_models.append( build_intermediate_model(init_shape, feature_model, build_mapping(init_shape=feature_shape, blocks = forward_blocks[:(l+1)]) ) )
  
  progressive_generator = build_generator(init_shape, feature_model, mapping_x2y, mapping_y2x, recon_model)
  progressive_generator.compile(optimizer=optimizer, loss='mse', loss_weights=lws) 
  progressive_generator.summary()
  
  for e in range(epochs):
    tmp_hist = progressive_generator.fit(x=x, y=y, batch_size=batch_size, 
                                         epochs=1, 
                                         shuffle=True, 
                                         validation_data=(x_test,y_test))
    
    save_figure(progressive_generator, t1_vis, t2_vis, output_path+prefix+'_current'+str(n_epochs)+'fig_patches.png')

    L.append( compute_lipschitz() )
    plt.figure(figsize=(4,4))    
    plt.plot(L)
    plt.savefig(output_path+prefix+'_lipschitz.png',dpi=150, bbox_inches='tight')
    plt.close()
    
    R.append( compute_reversibility_error() )
    plt.figure()
    plt.plot(R)
    plt.legend(['$\| f(x) - m^{-1} \circ m \circ f(x) \|^2$'])
    plt.savefig(output_path+prefix+'_inverr.png',dpi=150, bbox_inches='tight')
    plt.close()   

    nv = compute_norm_velocity()
    norm_velocity[0:len(nv),n_epochs] = nv
    plt.figure()    
    im = plt.imshow(norm_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()

    ngv = compute_gradient_velocity()
    norm_gradient_velocity[0:len(ngv),n_epochs] = ngv
    plt.figure()    
    im = plt.imshow(norm_gradient_velocity,cmap=plt.cm.gray,
                 interpolation="nearest")
    plt.colorbar(im)
    plt.savefig(output_path+prefix+'_norm_gradient_velocity_matrix.png',dpi=150, bbox_inches='tight')
    plt.close()


    n_epochs += 1
    
  
    if hist is None:
      hist = tmp_hist
    else:
      for k in list(hist.history.keys()):
        hist.history[k] = hist.history[k] + tmp_hist.history[k] 

#Save hist figure
plt.figure(figsize=(4,4))

plt.plot(hist.history['loss'])
plt.plot(hist.history['model_1_loss'])
for i in range(7):
  plt.plot(hist.history['model_1_'+str(i+1)+'_loss'])

plt.ylabel('training loss')  
plt.xlabel('epochs')
plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'], loc='upper right')
plt.savefig(output_path+prefix+'_loss_training.png',dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(4,4))

plt.plot(hist.history['val_loss'])
plt.plot(hist.history['val_model_1_loss'])
for i in range(7):
  plt.plot(hist.history['val_model_1_'+str(i+1)+'_loss'])

plt.ylabel('validation loss')  
plt.xlabel('epochs')
plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'], loc='upper right')
plt.savefig(output_path+prefix+'_loss_validation.png',dpi=150, bbox_inches='tight')
plt.close()

#%%    
#Pickle / save everything
import copy 

joblib.dump((copy.deepcopy(hist.history), L, R, norm_velocity, norm_gradient_velocity),output_path+prefix+'_hist.pkl', compress=True)
feature_model.save(output_path+prefix+'_feature_model.h5')
mapping_x2y.save(output_path+prefix+'_mapping_x2y.h5')
mapping_y2x.save(output_path+prefix+'_mapping_y2x.h5')
recon_model.save(output_path+prefix+'_recon_model.h5')

#%%
from tensorflow.keras.models import load_model
fm = load_model(output_path+prefix+'_feature_model.h5')
mx2y = load_model(output_path+prefix+'_mapping_x2y.h5')
my2x = load_model(output_path+prefix+'_mapping_y2x.h5')
rm = load_model(output_path+prefix+'_recon_model.h5')

gen = build_generator(init_shape, fm, mx2y, my2x, rm)
gen.compile(optimizer=optimizer, loss='mse', loss_weights=lws) 
gen.summary()

save_figure(gen, t1_vis, t2_vis, output_path+prefix+'_check_final_epoch.png')

#%%

sys.exit()
    
#%%


# hist = paired_generator.fit(x=x, 
#                           y=y, 
#                           batch_size=batch_size, 
#                           epochs=epochs, 
#                           shuffle=True,
#                           callbacks=callbacks,
#                           validation_data=(x_test,y_test))

# #Save hist figure
# plt.figure(figsize=(4,4))

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['model_1_loss'])
# for i in range(7):
#   plt.plot(hist.history['model_1_'+str(i+1)+'_loss'])

# plt.ylabel('training loss')  
# plt.xlabel('epochs')
# plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'], loc='upper right')
# plt.savefig(output_path+prefix+'_loss_training.png',dpi=150, bbox_inches='tight')
# plt.close()

# plt.figure(figsize=(4,4))

# plt.plot(hist.history['val_loss'])
# plt.plot(hist.history['val_model_1_loss'])
# for i in range(7):
#   plt.plot(hist.history['val_model_1_'+str(i+1)+'_loss'])

# plt.ylabel('validation loss')  
# plt.xlabel('epochs')
# plt.legend(['overall loss','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$','$r \circ f(x)$','$r \circ f(y)$'], loc='upper right')
# plt.savefig(output_path+prefix+'_loss_validation.png',dpi=150, bbox_inches='tight')
# plt.close()

#%%
sys.exit()

#%%

#%%


#%%

# ix = Input(shape=init_shape)	

# fx = feature_model(ix)   

# #Forward     
# mapping_x2y = build_mapping(init_shape=feature_shape, blocks = forward_blocks)
# mx2y = mapping_x2y(fx)
# rx2y = recon_model(mx2y)

# model = Model(inputs=[ix], 
#               outputs=[rx2y])

# model.compile(optimizer=optimizer, loss='mse') 

# model.fit(x=X_train, 
#                           y=Y_train, 
#                           batch_size=batch_size, 
#                           epochs=epochs, 
#                           shuffle=True)

#%%
#https://github.com/parasj/checkmate

#Not working yet, issue at pip install

# import checkmate
# from tqdm import tqdm

# train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

# from checkmate.tf2.wrapper import compile_tf2
# element_spec = train_ds.__iter__().__next__()
# train_iteration = compile_tf2(
#     paired_generator,
#     loss='mse',
#     optimizer=optimizer,
#     input_spec=element_spec[0],  # retrieve first element of dataset
#     label_spec=element_spec[1]
# )

# train_loss = tf.keras.metrics.Mean(name="train_loss")
# test_loss = tf.keras.metrics.Mean(name="test_loss")

# for epoch in range(10):
#     # Reset the metrics at the start of the next epoch
#     train_loss.reset_states()
#     test_loss.reset_states()

#     with tqdm(total=x[0].shape[0]) as pbar:
#         for images, labels in train_ds:
#             predictions, loss_value = train_iteration(images, labels)
#             train_loss(loss_value)
#             pbar.update(images.shape[0])
#             pbar.set_description('Train epoch {}; loss={:0.4f}'.format(epoch + 1, train_loss.result()))


#%%

#%%
sys.exit()

#%%
from model_zoo import conv_block_2d
from tensorflow.keras.layers import AveragePooling2D, Add, Subtract
from tensorflow.keras.utils import plot_model

def create_generator(n_filters):
  feature_shape = (None, None, n_filters)

  f_in = Input(shape=init_shape)	
  f_out = conv_block_2d(f_in, n_filters, bn=False, do=0)
  feature_model = Model(inputs=f_in, outputs=f_out)
  
  block_f_x2y = []
  block_g_x2y = []
    
  for l in range(n_layers):
    bf = build_block_model_2d(init_shape=half_feature_shape, 
                              n_filters=(int)(n_filters/2), 
                              n_layers=n_layers_residual, 
                              kernel_initializer=ki, 
                              activity_regularizer=ar)
    bg = build_block_model_2d(init_shape=half_feature_shape, 
                              n_filters=(int)(n_filters/2), 
                              n_layers=n_layers_residual, 
                              kernel_initializer=ki, 
                              activity_regularizer=ar)
    block_f_x2y.append(bf)
    block_g_x2y.append(bg)

  mapping_x2y = build_reversible_forward_model_2d(init_shape=feature_shape,
                                                  block_f=block_f_x2y, 
                                                  block_g = block_g_x2y, n_layers=n_layers) 
  
  mapping_y2x = build_reversible_backward_model_2d(init_shape=feature_shape,
                                                  block_f=block_f_x2y, 
                                                  block_g = block_g_x2y, n_layers=n_layers)

  r_in = Input(shape=feature_shape)	
  r_out = conv_block_2d(r_in, n_filters, bn=False, do=0)
  r_out = Conv2D(filters=n_channelsY, kernel_size=(3,3), padding='same', kernel_initializer='glorot_uniform',
              use_bias=True,
              strides=1)(r_out)
  recon_model = Model(inputs=r_in, outputs=r_out)

  ix = Input(shape=init_shape)	
  fx= feature_model(ix)
  mx= mapping_x2y(fx)
  rx= recon_model(mx)  
  g_x2y = Model(inputs=ix, outputs=rx)
  g_x2y.summary() 

  iy = Input(shape=init_shape)	
  fy= feature_model(iy)
  my= mapping_y2x(fy)
  ry= recon_model(my)  
  g_y2x = Model(inputs=iy, outputs=ry)
  g_y2x.summary() 
  
  return g_x2y, g_y2x

#%%
n_scales = 3
g_x2y = []
g_y2x = []
for i in range(n_scales):
  g = create_generator(n_filters)
  g_x2y.append( g[0] )
  g_y2x.append( g[1] )

#Scales : 
# 0 : original scale
# 1 : downsampled by factor 2^1
# 2 : downsampled by factor 2^2 = 4
  

def get_ms_generator(scales, gens):
  n_scales = len(scales)  
  ix = Input(shape=init_shape)	

  dx = [None] * n_scales
  rx = [None] * n_scales
  ux = [None] * n_scales
  
  for i in range(n_scales):
    dx[i] = AveragePooling2D(pool_size=(2**scales[i], 2**scales[i]), strides=None, padding='same')(ix)
    rx[i] = gens[scales[i]](dx[i])
    ux[i] = UpSampling2D(size=(2**scales[i], 2**scales[i]), data_format=None, interpolation='bilinear')(rx[i])
  
  if n_scales > 1:  
    ox = Add()(ux)
  else:
    ox = ux[0]
    
  g_scale = Model(inputs=ix, outputs=ox)  
  #g_scale.summary()
      
  g_scale.compile(loss='mse', optimizer=optimizer)  
  return g_scale

  
#plot_model(g_x2y_scale, to_file='g_x2y_scale.png')

#%%

def visual_check():
  n = 10
  step = int(X_train.shape[0]/10)
  x = X_train[0:n*step:step,:,:,:]
  y = Y_train[0:n*step:step,:,:,:]
  
  patch_list=[x,y]
  patch_list.append( get_ms_generator([*range(n_scales)], g_x2y)(x) )
  for i in range(n_scales):
    patch_list.append( get_ms_generator([i], g_x2y)(x) )
  
  n_rows = n 
  n_cols = len(patch_list)
  
    
  plt.figure(figsize=(2. * n_cols, 2. * n_rows))
  for i in range(n_rows):
    for j in range(n_cols):
      sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
      if j < 8:
        vmin = -2 
        vmax = 2
        sub.imshow(patch_list[j][i,:,:,0],
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=vmin,vmax=vmax)
      else:
        vmin = 0
        vmax = 1
        sub.imshow(patch_list[j][i] * np.ones((2,2)),
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=vmin,vmax=vmax)
      sub.axis('off')
  plt.axis('off')
  plt.show()
#%%
#Visual check
# g_x2y_scale = get_ms_generator([3], g_x2y)
# g_x2y_scale.fit(x=[X_train], y=[Y_train], batch_size=64, epochs=1, shuffle=True)
# visual_check()

# g_x2y_scale = get_ms_generator([2], g_x2y)
# g_x2y_scale.fit(x=[X_train], y=[Y_train], batch_size=64, epochs=1, shuffle=True)

# visual_check()

# g_x2y_scale = get_ms_generator([1,2], g_x2y)
# g_x2y_scale.fit(x=[X_train], y=[Y_train], batch_size=64, epochs=1, shuffle=True)

# visual_check()

# g_x2y_scale = get_ms_generator([0,1,2], g_x2y)
# g_x2y_scale.fit(x=[X_train], y=[Y_train], batch_size=64, epochs=1, shuffle=True)

# visual_check()  


#%%

def build_discriminator(init_shape):
  i = Input(shape=init_shape)	
  o = i 
  nf = n_filters_discriminator
  for level in range(n_levels_discriminator):
    o = Conv2D(nf, kernel_size=3, strides=2, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(discriminator_regularizer),
               activity_regularizer=regularizers.l1(discriminator_regularizer))(o)
    o = BatchNormalization()(o)
    nf = nf*2
  o = Conv2D(1, kernel_size=4,strides=1,padding='same', activation='linear')(o)
  o = GlobalAveragePooling2D()(o)
  #o = Activation('sigmoid')(o) #not used in wasserstein GAN -> linear act in conv layer
  return Model(i,o)

#Discriminator X domain
def metric(y_true,y_pred):
  return tf.keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)

d_X = build_discriminator(init_shape)
d_X.compile(loss=loss_discriminator,
      optimizer=optimizer,
      metrics=[metric])  
#Discriminator Y domain
d_Y = build_discriminator(init_shape)
d_Y.compile(loss=loss_discriminator,
      optimizer=optimizer,
      metrics=[metric])  

#%%
def save_samples(suffix):
  n = 10
  step = int(X_train.shape[0]/10)
  x = X_train[0:n*step:step,:,:,:]
  y = Y_train[0:n*step:step,:,:,:]

  current_gx = g_x2y_scale(x)
  current_gy = g_y2x_scale(y)

  current_idx = g_y2x_scale(x) 
  current_idy = g_x2y_scale(y)

  cycle_x = g_y2x_scale(current_gx)
  cycle_y = g_x2y_scale(current_gy)

  dx_real = np.reshape(d_X.predict(x),-1) 
  dx_fake = np.reshape(d_X.predict(current_gy),-1) 

  print(dx_fake)
  print(dx_real)

  dy_real = np.reshape(d_Y.predict(y),-1) 
  dy_fake = np.reshape(d_Y.predict(current_gx),-1) 

  titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$', 'dx real', 'dx fake', 'dy real', 'dy fake']

  patch_list=[x,y,current_gx,current_gy,current_idx,current_idy,cycle_x,cycle_y,dx_real, dx_fake,dy_real, dy_fake]  

  n_rows = n 
  n_cols = len(titles)

  
  plt.figure(figsize=(2. * n_cols, 2. * n_rows))
  for i in range(n_rows):
    for j in range(n_cols):
      sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
      if j < 8:
        vmin = -2 
        vmax = 2
        sub.imshow(patch_list[j][i,:,:,0],
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=vmin,vmax=vmax)
      else:
        vmin = 0
        vmax = 1
        sub.imshow(patch_list[j][i] * np.ones((2,2)),
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=vmin,vmax=vmax)
      sub.axis('off')
      if i == 0:
        sub.set_title(titles[j])
  plt.axis('off')
  plt.savefig(output_path+'/current_'+suffix+'.png',dpi=150, bbox_inches='tight')
  plt.close()
#%%

if pre_generator_training is True:
  scales = [] 
  
  for s in reversed(range(n_scales)):
    scales.append(s)
    print(scales)
    
    print('One supervised generative epoch')    
    img_X = Input(shape=init_shape)
    img_Y = Input(shape=init_shape)
  
    g_x2y_scale = get_ms_generator(scales, g_x2y)
    g_y2x_scale = get_ms_generator(scales, g_y2x)
    
    current_gx = g_x2y_scale(img_X)
    current_gy = g_y2x_scale(img_Y)
  
    cycle_x = g_y2x_scale(current_gx)
    cycle_y = g_x2y_scale(current_gy)
  
    g_joint = Model(inputs=[img_X, img_Y],
                      outputs=[ current_gx, current_gy, cycle_x, cycle_y])
    g_joint.compile(loss='mse',
                    optimizer=optimizer)    
  
    g_joint.fit(x=[X_train,Y_train], y=[Y_train,X_train,X_train,Y_train], batch_size=64, epochs=1, shuffle=True)  
  
    save_samples('gen_pretraining_scale_'+str(s))
    
#%%
valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

n_batches = int(X_train.shape[0]/batch_size)

print('Alternate learning')
#scales_list = [[3],[2,3],[1,2,3],[0,1,2,3]]
scales_list = [[2],[1,2],[0,1,2]]

for scales in scales_list:
  
  s_max = np.min(scales)
  
  #Model to downsample and upsample images
  ix = Input(shape=init_shape)	
  dx = AveragePooling2D(pool_size=(2**s_max, 2**s_max), strides=None, padding='same')(ix)
  ux = UpSampling2D(size=(2**s_max, 2**s_max), data_format=None, interpolation='bilinear')(dx)  
  down_and_up = Model(ix,ux)
  
  g_x2y_scale = get_ms_generator(scales, g_x2y)
  g_y2x_scale = get_ms_generator(scales, g_y2x)
  
  #Not trainable in the combiend model
  d_X.trainable = False
  d_Y.trainable = False

  #Combined model
  img_X = Input(shape=init_shape)
  img_Y = Input(shape=init_shape)
  
  fake_Y = g_x2y_scale(img_X)
  fake_X = g_y2x_scale(img_X)
  
  cycle_X = g_y2x_scale(fake_Y)
  cycle_Y = g_x2y_scale(fake_X)
  
  # Discriminators determines validity of translated images
  valid_X = d_X(fake_X)
  valid_Y = d_Y(fake_Y)
  
  # Combined model trains generators to fool discriminators
  combined = Model(inputs=[img_X, img_Y],
                    outputs=[ valid_X, valid_Y,
                              cycle_X, cycle_Y])
  combined.compile(loss=[loss_discriminator, loss_discriminator,
                          'mae', 'mae'],
                    loss_weights=[lambda_adv, lambda_adv,
                                lambda_cycle, lambda_cycle],
                    optimizer=optimizer)


 
  for epoch in range(epochs):
    print('Epoch : '+str(epoch))
    for batch_i in range(n_batches):
      
      if batch_i % save_interval == 0:
        save_samples(prefix+'scale'+str(s_max)+'_epoch'+str(epoch))
      
      
      for _ in range(n_critic_steps):
        # Generate a batch of new images
        id_imgs = np.random.randint(0, X_train.shape[0], batch_size)
        imgs_X = X_train[id_imgs]
        imgs_Y = Y_train[id_imgs]
    
        # Translate images to opposite domain
        fake_X = g_y2x_scale.predict(imgs_Y[:int(batch_size/2)])
        fake_Y = g_x2y_scale.predict(imgs_X[:int(batch_size/2)])
        
        # Process images for multi-scale discriminator
        imgs_X = down_and_up(imgs_X)
        imgs_Y = down_and_up(imgs_Y)
  
        # Train the discriminators (original images = real / translated = Fake)
    
        dx_data_train = np.concatenate([imgs_X[:int(batch_size/2)],fake_X],axis=0)
        dx_label_train = np.concatenate([valid[:int(batch_size/2)],fake[:int(batch_size/2)]],axis=0)
        dX_loss = d_X.train_on_batch(dx_data_train,dx_label_train)
  
        dy_data_train = np.concatenate([imgs_Y[:int(batch_size/2)],fake_Y],axis=0)
        dy_label_train = np.concatenate([valid[:int(batch_size/2)],fake[:int(batch_size/2)]],axis=0)
        dY_loss = d_Y.train_on_batch(dy_data_train,dy_label_train)
  
        # Total disciminator loss 
        d_loss = 0.5 * np.add(dX_loss, dY_loss)
  
      # Train the generators
      g_loss = combined.train_on_batch([imgs_X, imgs_Y],
                                      [valid, valid,
                                      imgs_X, imgs_Y])

      # Plot the progress
      print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, cycle: %05f] " \
          % ( epoch, epochs,
              batch_i, n_batches,
              d_loss[0], 100*d_loss[1],
              g_loss[0],
              np.mean(g_loss[1:3]),
              np.mean(g_loss[3:5])))



#%%
sys. exit()
#%%

#%%


print('Feature model')
#feature_model = build_encoder_2d(init_shape=init_shape, n_filters=n_filters, n_levels = n_levels)
feature_model = UNet(img_shape = init_shape, out_ch=n_filters, start_ch = 8, depth = 3, last_activation = 'tanh')
feature_model.summary()

 
#recon_model = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = n_levels)
recon_model = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = 0)

block_f_x2y = []
block_g_x2y = []

for l in range(n_layers):
  bf = build_block_model_2d(init_shape=half_feature_shape, 
                            n_filters=(int)(n_filters/2), 
                            n_layers=n_layers_residual, 
                            kernel_initializer=ki, 
                            activity_regularizer=ar)
  bg = build_block_model_2d(init_shape=half_feature_shape, 
                            n_filters=(int)(n_filters/2), 
                            n_layers=n_layers_residual, 
                            kernel_initializer=ki, 
                            activity_regularizer=ar)
  block_f_x2y.append(bf)
  block_g_x2y.append(bg)

mapping_x2y = build_reversible_forward_model_2d(init_shape=feature_shape,
                                                block_f=block_f_x2y, 
                                                block_g = block_g_x2y, n_layers=n_layers)
mapping_y2x = build_reversible_backward_model_2d(init_shape=feature_shape,
                                                  block_f=block_f_x2y, 
                                                  block_g = block_g_x2y, n_layers=n_layers)

#Generator : g_x2y
ix = Input(shape=init_shape)	
fx= feature_model(ix)
mx= mapping_x2y(fx)
rx= recon_model(mx)
g_x2y = Model(inputs=ix, outputs=rx)

#Generator : g_y2x
iy = Input(shape=init_shape)	
fy= feature_model(iy)
my= mapping_y2x(fy)
ry= recon_model(my)
g_y2x = Model(inputs=iy, outputs=ry)

print('Mapping')
mapping_x2y.summary()
print('Reconstruction Model')
recon_model.summary()

print('Generator')
g_x2y.summary()

def build_discriminator(init_shape):
  i = Input(shape=init_shape)	
  o = i 
  nf = n_filters_discriminator
  for level in range(n_levels_discriminator):
    o = Conv2D(nf, kernel_size=3, strides=2, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(discriminator_regularizer),
               activity_regularizer=regularizers.l1(discriminator_regularizer))(o)
    o = BatchNormalization()(o)
    nf = nf*2
  o = Conv2D(1, kernel_size=4,strides=1,padding='same', activation='linear')(o)
  o = GlobalAveragePooling2D()(o)
  #o = Activation('sigmoid')(o) #not used in wasserstein GAN -> linear act in conv layer
  return Model(i,o)

#Discriminator X domain
def metric(y_true,y_pred):
  return tf.keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)


d_X = build_discriminator(init_shape)
d_X.compile(loss=loss_discriminator,
      optimizer=optimizer,
      metrics=[metric])
#Discriminator Y domain
d_Y = build_discriminator(init_shape)
d_Y.compile(loss=loss_discriminator,
      optimizer=optimizer,
      metrics=[metric])  


print('Discriminator')
d_X.summary()

#Not trainable in the combiend model
d_X.trainable = False
d_Y.trainable = False     

#Combined model
img_X = Input(shape=init_shape)
img_Y = Input(shape=init_shape)

fake_Y = g_x2y(img_X)
fake_X = g_y2x(img_Y)

cycle_X = g_y2x(fake_Y)
cycle_Y = g_x2y(fake_X)

id_X = g_y2x(img_X)
id_Y = g_x2y(img_Y)

ae_X = feature_model(img_X)
ae_X = recon_model(ae_X)

ae_Y = feature_model(img_Y)
ae_Y = recon_model(ae_Y)


# Discriminators determines validity of translated images
valid_X = d_X(fake_X)
valid_Y = d_Y(fake_Y)

# Combined model trains generators to fool discriminators
combined = Model(inputs=[img_X, img_Y],
                  outputs=[ valid_X, valid_Y,
                            cycle_X, cycle_Y,
                            id_X, id_Y,
                            ae_X, ae_Y])
combined.compile(loss=[loss_discriminator, loss_discriminator,
                        'mae', 'mae',
                        'mae', 'mae',
                        'mae','mae'],
                  loss_weights=[lambda_adv, lambda_adv,
                              lambda_cycle, lambda_cycle,
                              lambda_id, lambda_id,
                              lambda_ae, lambda_ae ],
                  optimizer=optimizer)    

#%%
#--------------------------------------------------------------------------
def save_samples(suffix):
  n = 10
  step = int(X_train.shape[0]/10)
  x = X_train[0:n*step:step,:,:,:]
  y = Y_train[0:n*step:step,:,:,:]

  current_gx = g_x2y(x)
  current_gy = g_y2x(y)

  current_idx = g_y2x(x) 
  current_idy = g_x2y(y)

  cycle_x = g_y2x(current_gx)
  cycle_y = g_x2y(current_gy)

  dx_real = np.reshape(d_X.predict(x),-1) 
  dx_fake = np.reshape(d_X.predict(current_gy),-1) 

  print(dx_fake)
  print(dx_real)

  dy_real = np.reshape(d_Y.predict(y),-1) 
  dy_fake = np.reshape(d_Y.predict(current_gx),-1) 

  titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$', 'dx real', 'dx fake', 'dy real', 'dy fake']

  patch_list=[x,y,current_gx,current_gy,current_idx,current_idy,cycle_x,cycle_y,dx_real, dx_fake,dy_real, dy_fake]  

  n_rows = n 
  n_cols = len(titles)

  
  plt.figure(figsize=(2. * n_cols, 2. * n_rows))
  for i in range(n_rows):
    for j in range(n_cols):
      sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
      if j < 8:
        vmin = -2 
        vmax = 2
        sub.imshow(patch_list[j][i,:,:,0],
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=vmin,vmax=vmax)
      else:
        vmin = 0
        vmax = 1
        sub.imshow(patch_list[j][i] * np.ones((2,2)),
                    cmap=plt.cm.gray,
                    interpolation="nearest",
                    vmin=vmin,vmax=vmax)
      sub.axis('off')
      if i == 0:
        sub.set_title(titles[j])
  plt.axis('off')
  plt.savefig(output_path+'/current_'+suffix+'.png',dpi=150, bbox_inches='tight')
  plt.close()
#--------------------------------------------------------------------------

#valid = np.ones((batch_size,patch_size_discriminator,patch_size_discriminator,1))
#fake = np.zeros((batch_size,patch_size_discriminator,patch_size_discriminator,1))
valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

print('Valid shape')
print(valid.shape)  

n_batches = int(X_train.shape[0]/batch_size)

if pre_generator_training is True:
  print('One supervised generative epoch')  
  img_X = Input(shape=init_shape)
  img_Y = Input(shape=init_shape)

  current_gx = g_x2y(img_X)
  current_gy = g_y2x(img_Y)

  cycle_x = g_y2x(current_gx)
  cycle_y = g_x2y(current_gy)

  id_X = g_y2x(img_X)
  id_Y = g_x2y(img_Y)

  ae_X = feature_model(img_X)
  ae_X = recon_model(ae_X)

  ae_Y = feature_model(img_Y)
  ae_Y = recon_model(ae_Y)

  g_joint = Model(inputs=[img_X, img_Y],
                    outputs=[ current_gx, current_gy, cycle_x, cycle_y, id_X, id_Y, ae_X, ae_Y])
  g_joint.compile(loss='mse',
                  optimizer=optimizer)    

  g_joint.fit(x=[X_train,Y_train], y=[Y_train,X_train,X_train,Y_train,X_train,Y_train,X_train,Y_train], batch_size=16, epochs=1, shuffle=True)  

  save_samples('gen_pretraining')

if pre_discriminator_training is True:
  print('5 Epochs for discriminators alone')
  for _ in range(5):  
    for batch_i in range(int(X_train.shape[0]/32)):
      id_imgs = np.random.randint(0, X_train.shape[0], batch_size)
      imgs_X = X_train[id_imgs]
      imgs_Y = Y_train[id_imgs]
      fake_Y = g_x2y.predict(imgs_X[:int(batch_size/2)])
      fake_X = g_y2x.predict(imgs_Y[:int(batch_size/2)])
      dx_data_train = np.concatenate([imgs_X[:int(batch_size/2)],fake_X],axis=0)
      dx_label_train = np.concatenate([valid[:int(batch_size/2)],fake[:int(batch_size/2)]],axis=0)
      dX_loss = d_X.train_on_batch(dx_data_train,dx_label_train)
      dy_data_train = np.concatenate([imgs_Y[:int(batch_size/2)],fake_Y],axis=0)
      dy_label_train = np.concatenate([valid[:int(batch_size/2)],fake[:int(batch_size/2)]],axis=0)
      dY_loss = d_Y.train_on_batch(dy_data_train,dy_label_train)

  save_samples('dis_pretraining')



print('Alternate learning')
for epoch in range(epochs):
  print('Epoch : '+str(epoch))
  for batch_i in range(n_batches):
    for _ in range(n_critic_steps):
      # Generate a batch of new images
      id_imgs = np.random.randint(0, X_train.shape[0], batch_size)
      imgs_X = X_train[id_imgs]
      imgs_Y = Y_train[id_imgs]
  
      # Translate images to opposite domain
      fake_Y = g_x2y.predict(imgs_X[:int(batch_size/2)])
      fake_X = g_y2x.predict(imgs_Y[:int(batch_size/2)])

      # Train the discriminators (original images = real / translated = Fake)

      dx_data_train = np.concatenate([imgs_X[:int(batch_size/2)],fake_X],axis=0)
      dx_label_train = np.concatenate([valid[:int(batch_size/2)],fake[:int(batch_size/2)]],axis=0)
      dX_loss = d_X.train_on_batch(dx_data_train,dx_label_train)

      # dX_loss_real = d_X.train_on_batch(imgs_X, valid)
      # dX_loss_fake = d_X.train_on_batch(fake_X, fake)
      # dX_loss = 0.5 * np.add(dX_loss_real, dX_loss_fake)

      dy_data_train = np.concatenate([imgs_Y[:int(batch_size/2)],fake_Y],axis=0)
      dy_label_train = np.concatenate([valid[:int(batch_size/2)],fake[:int(batch_size/2)]],axis=0)
      dY_loss = d_Y.train_on_batch(dy_data_train,dy_label_train)

      # dY_loss_real = d_Y.train_on_batch(imgs_Y, valid)
      # dY_loss_fake = d_Y.train_on_batch(fake_Y, fake)
      # dY_loss = 0.5 * np.add(dY_loss_real, dY_loss_fake)

      # Total disciminator loss 
      d_loss = 0.5 * np.add(dX_loss, dY_loss)

      if clipping is True:
        # Clip critic weights
        for l in d_X.layers:
          weights = l.get_weights()
          weights = [np.clip(w, -clip_value, clip_value) for w in weights]
          l.set_weights(weights)

        for l in d_Y.layers:
          weights = l.get_weights()
          weights = [np.clip(w, -clip_value, clip_value) for w in weights]
          l.set_weights(weights)


    # Train the generators
    g_loss = combined.train_on_batch([imgs_X, imgs_Y],
                                    [valid, valid,
                                    imgs_X, imgs_Y,
                                    imgs_X, imgs_Y,
                                    imgs_X, imgs_Y])

    # Plot the progress
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, cycle: %05f, id: %05f, ae: %05f] " \
        % ( epoch, epochs,
            batch_i, n_batches,
            d_loss[0], 100*d_loss[1],
            g_loss[0],
            np.mean(g_loss[1:3]),
            np.mean(g_loss[3:5]),
            np.mean(g_loss[5:7]),
            np.mean(g_loss[7:9])))

    if batch_i % save_interval == 0:
      #save_samples('epoch'+str(epoch)+'_batch'+str(batch_i))
      save_samples(prefix+'epoch'+str(epoch))

      
