#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:00:03 2019

@author: rousseau
"""

# SSSISR : self-supervised single-image super resolution

import os
import sys
print(sys.platform)
if sys.platform == 'darwin':
  os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" 
  
import argparse
import nibabel
import numpy as np
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state
from keras import backend as K
from scipy import signal
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from keras.layers import GlobalAveragePooling2D, Multiply, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers

def get_patches(imx, imy, patch_shape, max_patches=1000):
  X = None
  Y = None
  random_state = None

  for j in range(imx.shape[2]): #loop over the slices
    # X : ilr (interpolated low resolution), Y : arr (high resolution)
    pX = extract_patches(imx[:,:,j,:], patch_shape, extraction_step = 1)
    pY = extract_patches(imy[:,:,j,:], patch_shape, extraction_step = 1)
        
    rng = check_random_state(random_state)
    i_s = rng.randint(imx.shape[0] - patch_shape[0] + 1, size = max_patches)
    j_s = rng.randint(imx.shape[1] - patch_shape[1] + 1, size = max_patches)
    
    pX = pX[i_s, j_s,:]
    pY = pY[i_s, j_s,:]
    
    #Channel last
    pX = pX.reshape(-1, patch_shape[0], patch_shape[1],patch_shape[2])
    pY = pY.reshape(-1, patch_shape[0], patch_shape[1],patch_shape[2])

    if K.image_data_format() == 'channels_first':
      pX = np.moveaxis(pX, -1, 1)
      pY = np.moveaxis(pY, -1, 1)  
      
    if X is None:
      X = np.copy(pX)
      Y = np.copy(pY)
    else:
      X = np.concatenate((pX,X),axis=0)
      Y = np.concatenate((pY,Y),axis=0)  
  return (X,Y)

def get_model(n_filters=16,
              kernel_size=3,
              n_layers=10,
              use_attention=0,
              activity_regularizer=None):
  #Training a dedicated network
  if K.image_data_format() == 'channels_first':
    inputs = Input(shape=(n_channels, None, None))
    channel_axis = 1
  else:
    inputs = Input(shape=(None, None, n_channels))
    channel_axis = -1

    
  features = inputs
  features = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
                    use_bias=False,
                    strides=1,
                    activity_regularizer=activity_regularizer)(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('tanh')(features)
  
  mapping = features
  for l in range(n_layers):
    v = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
                      use_bias=False,
                      strides=1,
                      activity_regularizer=activity_regularizer)(mapping)
    v = Activation('relu')(v)
    v = Conv2D(n_filters, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
                      use_bias=False,
                      strides=1,
                      activity_regularizer=activity_regularizer)(v)
    
    if use_attention == 1:
      att = GlobalAveragePooling2D()(v)
      if K.image_data_format() == 'channels_first':
        att = Reshape((np.int(n_filters),1,1))(att)
      else:
        att = Reshape((1,1,np.int(n_filters)))(att)
        
      att = Conv2D(np.int(n_filters/2), (1,1), padding='same', kernel_initializer='he_normal',
                      use_bias=False,
                      strides=1,
                      activity_regularizer=activity_regularizer)(att)
      
      att = Activation('relu')(att)
      att = Conv2D(n_filters, (1,1), padding='same', kernel_initializer='he_normal',
                      use_bias=False,
                      strides=1,
                      activity_regularizer=activity_regularizer)(att)
      att = Activation('softmax')(att)
      v = Multiply()([att,v])
    
    mapping = Add()([mapping,v])
  
  #mapping = Add()([mapping,features])
  
  recon = mapping
  recon = Conv2D(n_channels, (kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
                    use_bias=False,
                    strides=1,
                    activity_regularizer=activity_regularizer)(recon)
  recon = Activation('linear')(recon)  

  recon = Add()([recon,inputs])

  model = Model(inputs=inputs, outputs=recon)
  return model

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', help='Input Image', type=str, required=True) 
  parser.add_argument('-s', '--simu', help='Simulated blur-down-up Image', type=str, required=True)
  parser.add_argument('-t', '--test', help='Upsampled Image', type=str, required=True)
  parser.add_argument('-m', '--model', help='Input Model', type=str, required=False)  
  parser.add_argument('-o', '--output', help='Output Prefix Image', type=str, required=True)
  parser.add_argument('-v', '--verbose', help='Verbose (0 or 1)', type=int, default=1, required=False)
  
  parser.add_argument('-l', '--layers', help='Number of layers', type=int, default=10, required=False)
  parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=50, required=False)
  parser.add_argument('-f', '--filters', help='Number of convolutional filters', type=int, default=64, required=False)
  parser.add_argument('-k', '--kernel_size', help='Conv kernel size', type=int, default=3, required=False)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, default=32, required=False)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, default=40, required=False)
  parser.add_argument(      '--max_patch', help='Max number of patches per image', type=int, default=3000, required=False)
  parser.add_argument('-w', '--windowing', help='Patch windowing (yes (1) or no (0) )', type=int, default=1, required=False)
  parser.add_argument(      '--act_reg', help='Activity L1 Regularization', type=float, default=0, required=False)
  parser.add_argument(      '--learning_rate', help='Learning rate', type=float, default=0.0001, required=False)

  args = parser.parse_args()

  print(K.image_data_format()) 
  
  #Sauver le modele
  #Mettre les hyperparametres en arguments
  #Hyper-parameters
  patch_size = args.patch_size # 50 may be the max size to get occipital brain boundaries in patch center
  max_patches = args.max_patch
  windowing = args.windowing 
  batch_size = args.batch_size #Chi-Hieu : 16
  n_filters = args.filters
  kernel_size = args.kernel_size
  n_layers = args.layers
  use_attention = 0
  learning_rate = args.learning_rate # Chi-Hieu : 0.0001
  loss = 'mae'
  epochs = args.epochs
  self_ensemble = 1
  act_reg = args.act_reg
  activity_regularizer=regularizers.l1(args.act_reg) # or None

  suffix = ''
  suffix += '_'+str(sys.platform)
  suffix += '_nl'+str(n_layers)
  suffix += '_nf'+str(n_filters)
  suffix += '_lr'+str(learning_rate)
  suffix += '_e'+str(epochs)
  suffix += '_bs'+str(batch_size)
  suffix += '_ps'+str(patch_size)
  suffix += '_mp'+str(max_patches)
  suffix += '_ar'+str(act_reg)
  if windowing == 1:
    suffix += '_windowed'
  if self_ensemble == 1:
    suffix += '_se'
  if use_attention == 1:
    suffix+= '_attention'
    
  fileout = args.output+suffix+'.nii.gz'
  print(fileout)
  
  #Read data
  imx = nibabel.load(args.simu)
  print(imx.header.get_zooms())  
  imx = imx.get_data().astype(float)
  print(imx.shape)

  imy = nibabel.load(args.input).get_data().astype(float)
  
  print(len(imx.shape))
  
  print('Assume that data format is X Y Z channels (channel last)')
  #Convert to 4D if necessary 
  if len(imx.shape) < 4:
    imx = np.expand_dims(imx,axis=-1)
  if len(imy.shape) < 4:
    imy = np.expand_dims(imy,axis=-1)

  print(imx.shape)
  print(imy.shape)
  
  n_channels = imx.shape[3]
  patch_shape = (patch_size,patch_size,n_channels)
  print(patch_shape)
  
  print('Extracting patches for training')
  X, Y = get_patches(imx, imy, patch_shape, max_patches)
  print(X.shape)

  if windowing == 1:
  
    w1d = signal.hamming(patch_size)
    w2d = np.sqrt(np.outer(w1d,w1d))
    w2d = np.expand_dims(w2d,axis=-1)
    
    X = X * w2d
    Y = Y * w2d
  
  data_gen_args = dict(horizontal_flip = True,
                         vertical_flip = True)#,
  
  X_datagen = ImageDataGenerator(**data_gen_args)
  Y_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
  seed = 1  
  X_generator = X_datagen.flow(X, batch_size=batch_size, seed=seed)
  Y_generator = Y_datagen.flow(Y, batch_size=batch_size, seed=seed)
  train_generator = zip(X_generator, Y_generator)  

  model = get_model(n_filters=n_filters,
              kernel_size=kernel_size,
              n_layers=n_layers,
              use_attention=use_attention,
              activity_regularizer=activity_regularizer)  
  model.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
  if args.verbose == 1:
    model.summary()  
  
  print('Learning...')
  hist = model.fit_generator(train_generator, 
                               steps_per_epoch=len(X) / batch_size, 
                               epochs = epochs, 
                               verbose=args.verbose, 
                               shuffle=True)   
  del X,Y,imx,imy

  print('Computing the high-resolution image')
  test_array = nibabel.load(args.test).get_data().astype(float)

  if len(test_array.shape) < 4:
    test_array = np.expand_dims(test_array,axis=-1)

        
  out1 = np.zeros(test_array.shape)
  out2 = np.zeros(test_array.shape)

  print(test_array.shape)
    
  if K.image_data_format() == 'channels_first' :
    print('Swapping axis because channel first mode is used in Keras')
    test_array = np.moveaxis(test_array, -1, 1)   
  
  
  for s in range(test_array.shape[0]): #Loop on coronal data
    #Self ensemble @ testing using flipped slice
    s1 = np.expand_dims(test_array[s,:,:,:],axis=0)
    r1 = model.predict(s1,batch_size=1)
    
    if self_ensemble == 1:
      a1 = 1
      a2 = 2
      if K.image_data_format() == 'channels_first':
        a1 = 3
      
      s2 = np.flip(s1,axis=a1)
      r2 = model.predict(s2,batch_size=1)
      r2 = np.flip(r2,axis=a1)
      
      s3 = np.flip(s2,axis=a2)
      r3 = model.predict(s3,batch_size=1)
      r3 = np.flip(r3,axis=a1)
      r3 = np.flip(r3,axis=a2)
  
      s4 = np.flip(s1,axis=a2)
      r4 = model.predict(s4,batch_size=1)
      r4 = np.flip(r4,axis=a2)
  
      #Ici probleme si channel first ?        
      out1[s,:,:,:] = (r1[0,:,:,:] + r2[0,:,:,:] + r3[0,:,:,:] + r4[0,:,:,:]) / 4.0
    else:
      out1[s,:,:,:] = r1[0,:,:,:]
      
  for s in range(test_array.shape[1]): #Loop on sagital data
    
    #Self ensemble @ testing using flipped slice
    s1 = np.expand_dims(test_array[:,s,:,:],axis=0)
    r1 = model.predict(s1,batch_size=1)
    
    if self_ensemble == 1:
      a1 = 0
      a2 = 2
      if K.image_data_format() == 'channels_first':
        a1 = 3
      
      s2 = np.flip(s1,axis=a1)
      r2 = model.predict(s2,batch_size=1)
      r2 = np.flip(r2,axis=a1)
      
      s3 = np.flip(s2,axis=a2)
      r3 = model.predict(s3,batch_size=1)
      r3 = np.flip(r3,axis=a1)
      r3 = np.flip(r3,axis=a2)
  
      s4 = np.flip(s1,axis=a2)
      r4 = model.predict(s4,batch_size=1)
      r4 = np.flip(r4,axis=a2)
  
      #Ici probleme si channel first ?        
      out2[:,s,:,:] = (r1[0,:,:,:] + r2[0,:,:,:] + r3[0,:,:,:] + r4[0,:,:,:]) / 4.0
    else:
      out2[:,s,:,:] = r1[0,:,:,:]
    
  out12 = 0.5*(out1+out2)    
  nibabel.save(nibabel.Nifti1Image(out12, nibabel.load(args.test).affine),fileout)    
