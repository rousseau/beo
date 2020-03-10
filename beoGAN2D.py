#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
from os.path import expanduser
home = expanduser("~")

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from model_zoo import build_encoder_2d, build_decoder_2d, build_block_model_2d, build_reversible_forward_model_2d, build_reversible_backward_model_2d, UNet


dataset = 'HCP' #HCP or IXI or Bazin
patch_size = 128
batch_size = 32  
epochs = 25
n_filters = 32
n_levels = 1          #for encoder and decoder to extract features
n_channelsX = 1
n_layers = 5          #number of layers for the numerical integration scheme (ResNet)
n_layers_residual = 1  #number of layers for the parametrization of the velocity fields
n_levels_discriminator = 4
n_filters_discriminator = 16
patch_size_discriminator = int(patch_size / 2**n_levels_discriminator)
ki = 'glorot_normal' #kernel initializer
ar = 0#1e-6                #activity regularization (L2)
optimizer = Adam(0.0002, 0.5)
lambda_cycle = 10
lambda_id = 0.1 * lambda_cycle 
save_interval=500

output_path = home+'/Sync/Experiments/'+dataset+'/'

if dataset == 'HCP':  
  X_train = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_training.pkl')
  Y_train = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_training.pkl')
  #X_test = joblib.load(home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_testing.pkl')
  #Y_test = joblib.load(home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_testing.pkl')
if dataset == 'Bazin':
  X_train = joblib.load(home+'/Sync/Experiments/Bazin/BlockFace_'+str(patch_size)+'_training.pkl')
  Y_train = joblib.load(home+'/Sync/Experiments/Bazin/Thio_'+str(patch_size)+'_training.pkl')


if K.image_data_format() == 'channels_first':
  init_shape = (n_channelsX, None, None)
  feature_shape = (n_filters, None, None)
  half_feature_shape = ((int)(n_filters/2), None, None)
  channel_axis = 1
else:
  #init_shape = (None, None,n_channelsX)
  init_shape = (patch_size, patch_size,n_channelsX)
  feature_shape = (None, None, n_filters)
  half_feature_shape = (None, None, (int)(n_filters/2))
  channel_axis = -1  

print(K.image_data_format())

#strategy = tf.distribute.MirroredStrategy()
#import subprocess
#n_gpu = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
#batch_size = batch_size * n_gpu
#print('Adapted batch size wrt number of GPUs : '+str(batch_size))
#print('Number of GPUs used:'+str(strategy.num_replicas_in_sync))

#with strategy.scope():
print('Feature model')
#feature_model = build_encoder_2d(init_shape=init_shape, n_filters=n_filters, n_levels = n_levels)
feature_model = UNet(img_shape = init_shape, out_ch=n_filters, start_ch = 8, depth = 3, last_activation = 'tanh')
feature_model.summary()

 
#recon_model = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = n_levels)
recon_model = build_decoder_2d(init_shape=feature_shape, n_filters=n_filters, n_levels = 0)

block_f_x2y = []
block_g_x2y = []

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
for l in range(n_layers):
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

print('Generator')
g_x2y.summary()

def build_discriminator(init_shape):
  i = Input(shape=init_shape)	
  o = i 
  nf = n_filters_discriminator
  for level in range(n_levels_discriminator):
    o = Conv2D(nf, kernel_size=3, strides=2, padding='same', activation='relu')(o)
    o = BatchNormalization()(o)
    nf = nf*2
  o = Conv2D(1, kernel_size=4,strides=1,padding='same')(o)
  return Model(i,o)

#Discriminator X domain
d_X = build_discriminator(init_shape)
d_X.compile(loss='mse',
      optimizer=optimizer,
      metrics=['accuracy'])
#Discriminator Y domain
d_Y = build_discriminator(init_shape)
d_Y.compile(loss='mse',
      optimizer=optimizer,
      metrics=['accuracy'])  


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
combined.compile(loss=['mse', 'mse',
                        'mae', 'mae',
                        'mae', 'mae',
                        'mae','mae'],
                  # loss_weights=[  1, 1,
                  #             lambda_cycle, lambda_cycle,
                  #             lambda_id, lambda_id ],
                  optimizer=optimizer)    
#--------------------------------------------------------------------------
def save_samples(suffix):
  n = 10
  step = 1000
  x = X_train[0:n*step:step,:,:,:]
  y = Y_train[0:n*step:step,:,:,:]

  current_gx = g_x2y(x)
  current_gy = g_y2x(y)

  current_idx = g_y2x(x) 
  current_idy = g_x2y(y)

  cycle_x = g_y2x(current_gx)
  cycle_y = g_x2y(current_gy)

  titles = ['$x$','$y$','$g(x)$','$g^{-1}(y)$','$r \circ m^{-1} \circ f(x)$','$r \circ m \circ f(y)$','$g^{-1} \circ g(x)$','$g \circ g^{-1}(y)$']

  patch_list=[x,y,current_gx,current_gy,current_idx,current_idy,cycle_x,cycle_y]  

  n_rows = n 
  n_cols = len(titles)

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
  plt.savefig(output_path+'/current_'+suffix+'.png',dpi=150, bbox_inches='tight')
  plt.close()

#--------------------------------------------------------------------------

valid = np.ones((batch_size,patch_size_discriminator,patch_size_discriminator,1))
fake = np.zeros((batch_size,patch_size_discriminator,patch_size_discriminator,1))

print('Valid shape')
print(valid.shape) 

n_batches = int(X_train.shape[0]/batch_size)

# print('one supervised generative epoch') #faire joint model
# img_X = Input(shape=init_shape)
# img_Y = Input(shape=init_shape)

# fake_Y = g_x2y(img_X)
# fake_X = g_y2x(img_Y)

# Combined model trains generators to fool discriminators
# g_joint = Model(inputs=[img_X, img_Y],
#                   outputs=[ fake_Y, fake_X])
# g_joint.compile(loss=['mse', 'mse'],
#                   loss_weights=[  1, 1],
#                   optimizer=optimizer)    

# g_joint.fit(x=[X_train,Y_train], y=[Y_train,X_train], batch_size=batch_size, epochs=2, shuffle=True)  

# save_samples('g_joint')

print('Alternate learning')
for epoch in range(epochs):
  for batch_i in range(n_batches):
    # Generate a batch of new images
    id_imgs = np.random.randint(0, X_train.shape[0], batch_size)
    imgs_X = X_train[id_imgs]
    imgs_Y = Y_train[id_imgs]

    # Translate images to opposite domain
    fake_Y = g_x2y.predict(imgs_X)
    fake_X = g_y2x.predict(imgs_Y)
    
    # Train the discriminators (original images = real / translated = Fake)
    dX_loss_real = d_X.train_on_batch(imgs_X, valid)
    dX_loss_fake = d_X.train_on_batch(fake_X, fake)
    dX_loss = 0.5 * np.add(dX_loss_real, dX_loss_fake)

    dY_loss_real = d_Y.train_on_batch(imgs_Y, valid)
    dY_loss_fake = d_Y.train_on_batch(fake_Y, fake)
    dY_loss = 0.5 * np.add(dY_loss_real, dY_loss_fake)

    # Total disciminator loss 
    d_loss = 0.5 * np.add(dX_loss, dY_loss)


    # Train the generators
    g_loss = combined.train_on_batch([imgs_X, imgs_Y],
                                    [valid, valid,
                                    imgs_X, imgs_Y,
                                    imgs_X, imgs_Y,
                                    imgs_X, imgs_Y])

    # Plot the progress
    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, cycle: %05f, id: %05f] " \
        % ( epoch, epochs,
            batch_i, n_batches,
            d_loss[0], 100*d_loss[1],
            g_loss[0],
            np.mean(g_loss[1:3]),
            np.mean(g_loss[3:5]),
            np.mean(g_loss[5:6])))

    if batch_i % save_interval == 0:
      save_samples('epoch'+str(epoch)+'_batch'+str(batch_i))
