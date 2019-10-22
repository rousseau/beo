#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 10:53:32 2019

@author: rousseau
"""

import os
import sys
print(sys.platform)
if sys.platform == 'darwin':
  os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" 
  
import nibabel
from os.path import expanduser
import matplotlib.pyplot as plt
#%matplotlib auto

import numpy as np
from tqdm import tqdm

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from utils import array_normalization
print(K.image_data_format())

home = expanduser("~")
data_path = home+'/Sync/Data/Cerveau/Foetal/CRL_Fetal_Brain_Atlas_2017/'

mris = []
segs = []

for i in range(21,39):
  mris.append(nibabel.load(data_path+'STA'+str(i)+'_r05.nii.gz').get_data())
  segs.append(nibabel.load(data_path+'STA'+str(i)+'_cortex_r05.nii.gz').get_data())
  
#Resize : from 216x302x248 -> 256x256x248
for i in range(len(mris)):
  tmp = np.zeros((256,256,mris[i].shape[2]))
  tmp[20:236,:,:] = mris[i][:,20:276,:]
  
  mask = np.zeros(tmp.shape)
  mask[tmp>0] = 1
  T1_norm = array_normalization(X=tmp,M=mask,norm=0)
  
  mris[i] = tmp
  tmp = np.zeros((256,256,mris[i].shape[2]))
  tmp[20:236,:,:] = segs[i][:,20:276,:]
  segs[i] = tmp
  

#Data Normalization

#Extract 2D images from 3D arrays
mri_patches = None
seg_patches = None

for i in tqdm(range(len(mris))):
  
  #Swap axis

  pmri = np.moveaxis(mris[i],-1,0)
  pseg = np.moveaxis(segs[i],-1,0)
  
  #remove zero patches
  pM = pseg.reshape(pseg.shape[0],-1)
  index = np.sum(pM,axis=1)>0
  pmri = pmri[ index ]
  pseg = pseg[ index ]  
  
  pmri = np.expand_dims(pmri, axis=-1)
  pseg = np.expand_dims(pseg, axis=-1)
    
  if mri_patches is None:
    mri_patches = np.copy(pmri)
    seg_patches = np.copy(pseg)
  else:
    mri_patches = np.concatenate((mri_patches,pmri),axis=0)
    seg_patches = np.concatenate((seg_patches,pseg),axis=0)    
  
print(mri_patches.shape)
print(seg_patches.shape)

#n_cols = 10
#n_rows = 10
#
#plt.figure(figsize=(2. * n_cols, 2. * n_rows))  
#
#for i in range(n_cols*n_rows):
#  sub = plt.subplot(n_rows, n_cols, i+1)
#  sub.imshow(mri_patches[i,:,:,0],
#               cmap=plt.cm.gray,
#               interpolation="nearest")
#
#  sub.imshow(seg_patches[i,:,:,0],
#               cmap='jet',
#               alpha=0.5,
#               interpolation="nearest")
#plt.show(block=False)    

smooth = 1.
def dice_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_loss(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(K.abs(y_true_f * y_pred_f))
  sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f))
  jac = (intersection + smooth) / (sum_ - intersection + smooth)
  return (1 - jac) * smooth

def dice_coef_loss(y_true, y_pred):
  return -dice_coef(y_true, y_pred)
  
learning_rate = 1e-4
batch_size = 5
epochs = 1000
n_channels = 1
load_keras_model = 0

if load_keras_model == 1:
  model = load_model(home+'/Sync/fetal_unet.h5')
else:  
  inputs = Input((None, None, n_channels))
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
  #x = BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool1)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool2)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv3)
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
  
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(pool4)
  conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv5)
  
  up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv5), conv4], axis=3)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up6)
  conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv6)
  
  up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv6), conv3], axis=3)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up7)
  conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv7)
  
  up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv7), conv2], axis=3)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up8)
  conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv8)
  
  up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', kernel_initializer = 'he_normal')(conv8), conv1], axis=3)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(up9)
  conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(conv9)
  
  conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer = 'he_normal')(conv9)
  
  
  #x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(inputs)
  #x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
  #x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
  #x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
  #x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer = 'he_normal')(x)
  #conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer = 'he_normal')(x)
  
  model = Model(inputs=[inputs], outputs=[conv10])
  
  #Est-ce que ce dice_coef gere les dice flous? (mse ou dice ou 'binary_crossentropy' ?)
  #model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
  model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
  #Test jaccard 


data_gen_args = dict(rotation_range=45, 
                     width_shift_range=0.1, 
                     height_shift_range=0.1,
                     shear_range=0.2, 
                     zoom_range=[0.8,1.2], 
                     brightness_range=[0.8,1.2],
                     fill_mode='nearest', 
                     horizontal_flip = True,
                     vertical_flip = True,
                     featurewise_center=False, 
                     samplewise_center=False, 
                     featurewise_std_normalization=False, 
                     samplewise_std_normalization=False,
                     data_format= 'channels_last')
X_datagen = ImageDataGenerator(**data_gen_args)
Y_datagen = ImageDataGenerator(**data_gen_args)
seed = 1
X_generator = X_datagen.flow(mri_patches, shuffle=True, batch_size=batch_size, seed=seed)
Y_generator = Y_datagen.flow(seg_patches, shuffle=True, batch_size=batch_size, seed=seed)
train_generator = zip(X_generator, Y_generator)  

#x,y = next(train_generator)    
#n_cols = 8
#n_rows = 4
#
#plt.figure(figsize=(2. * n_cols, 2. * n_rows))  
#
#for i in range(n_cols*n_rows):
#  sub = plt.subplot(n_rows, n_cols, i+1)
#  sub.imshow(x[i,:,:,0],
#               cmap=plt.cm.gray,
#               interpolation="nearest")
#
#  sub.imshow(y[i,:,:,0],
#               cmap='jet',
#               alpha=0.5,
#               interpolation="nearest")
#plt.show(block=False)  

## fitting model
hist = model.fit_generator(train_generator, 
                           verbose=1, 
                           steps_per_epoch=mri_patches.shape[0]/batch_size,
                           epochs=epochs,
                           shuffle=True)
model.save(home+'/Sync/fetal_unet.h5')

output = np.zeros(mris[0].shape)
for k in range(mris[0].shape[2]):
  tmp = np.expand_dims(mris[0][:,:,k],axis=0)
  tmp = np.expand_dims(tmp,axis=-1)
  tmp2 = model.predict(tmp,batch_size=1)
  output[:,:,k] = tmp2[0,:,:,0]
nibabel.save(nibabel.Nifti1Image(np.squeeze(output), nibabel.load(data_path+'STA21_r05.nii.gz').affine),home+'/Sync/toto.nii.gz')  
print('max de output : ')
print(np.max(output))
   
nibabel.save(nibabel.Nifti1Image(np.squeeze(mris[0]), nibabel.load(data_path+'STA21_r05.nii.gz').affine),home+'/Sync/tutu.nii.gz')  
nibabel.save(nibabel.Nifti1Image(np.squeeze(segs[0]), nibabel.load(data_path+'STA21_r05.nii.gz').affine),home+'/Sync/tata.nii.gz')  


output = np.zeros(mris[15].shape)
for k in range(mris[15].shape[2]):
  tmp = np.expand_dims(mris[15][:,:,k],axis=0)
  tmp = np.expand_dims(tmp,axis=-1)
  tmp2 = model.predict(tmp,batch_size=1)
  output[:,:,k] = tmp2[0,:,:,0]
nibabel.save(nibabel.Nifti1Image(np.squeeze(output), nibabel.load(data_path+'STA21_r05.nii.gz').affine),home+'/Sync/toto_15.nii.gz')  
   
nibabel.save(nibabel.Nifti1Image(np.squeeze(mris[15]), nibabel.load(data_path+'STA21_r05.nii.gz').affine),home+'/Sync/tutu_15.nii.gz')  
nibabel.save(nibabel.Nifti1Image(np.squeeze(segs[15]), nibabel.load(data_path+'STA21_r05.nii.gz').affine),home+'/Sync/tata_15.nii.gz')  

