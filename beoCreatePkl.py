#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
from os.path import expanduser
home = expanduser("~")
from dataset import get_ixi_2dpatches, get_hcp_2dpatches, get_hcp_4darrays
import glob
import nibabel
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

patch_size = 64
dataset = 'HCP'
ratio_training = 0.75

if dataset == 'IXI':  
  n_patches = 1000 #per slice 
  (T1_2D,T2_2D,PD_2D) = get_ixi_2dpatches(patch_size = patch_size, n_patches = n_patches)
  joblib.dump(T1_2D,home+'/Exp/IXI_T1_2D_'+str(patch_size)+'_training.pkl', compress=True)
  joblib.dump(T2_2D,home+'/Exp/IXI_T2_2D_'+str(patch_size)+'_training.pkl', compress=True)
  joblib.dump(PD_2D,home+'/Exp/IXI_PD_2D_'+str(patch_size)+'_training.pkl', compress=True)

if dataset == 'HCP':  
  n_patches = 50 #per slice (25 for ps=128)
  
  (T1s,T2s,masks) = get_hcp_4darrays()
  n_images = len(T1s)
  print('Number of loaded images : '+str(n_images))
  n_training_images = (int)(ratio_training*n_images)
  print('Number of training images : '+str(n_training_images))
  
  T1_training = T1s[0:n_training_images]
  T2_training = T2s[0:(int)(ratio_training*n_images)]
  mask_training = masks[0:(int)(ratio_training*n_images)]    
  
  (T1_2D_training,T2_2D_training) = get_hcp_2dpatches(patch_size = patch_size, n_patches = n_patches, data=(T1s[0:n_training_images],T2s[0:n_training_images],masks[0:n_training_images]))    
  joblib.dump(T1_2D_training,home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_training.pkl', compress=True)
  joblib.dump(T2_2D_training,home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_training.pkl', compress=True)

  (T1_2D_testing,T2_2D_testing) = get_hcp_2dpatches(patch_size = patch_size, n_patches = n_patches, data=(T1s[n_training_images:n_images],T2s[n_training_images:n_images],masks[n_training_images:n_images]))    
  joblib.dump(T1_2D_testing,home+'/Exp/HCP_T1_2D_'+str(patch_size)+'_testing.pkl', compress=True)
  joblib.dump(T2_2D_testing,home+'/Exp/HCP_T2_2D_'+str(patch_size)+'_testing.pkl', compress=True)

if dataset == 'Bazin':
  directories = glob.glob(home+'/Sync/Data/Cerveau/Adulte/SuperResolution-Francois/coreg_stains/', recursive=True)
  all_files = []
  for d in directories:
    files = glob.glob(d+'*Thio*')
    all_files.extend(files)
  all_files.sort()

  block_face = nibabel.load(home+'/Sync/Data/Cerveau/Adulte/SuperResolution-Francois/insitu_03_152017-li-crop-sub2-med_sir-img_n4_inv_cor.nii.gz')
  bf_data  = np.float32(block_face.get_fdata())

  #crop block face data and shape it for tensorflow------------------------------
  bf_samples = bf_data[:,:,313:475]
  bf_samples = bf_samples[:,:,::6]
  bf_samples = bf_samples[:,:,0:len(all_files)]
  bf_samples = np.expand_dims(bf_samples,axis=0)
  bf_samples = np.swapaxes(bf_samples,0,-1)

  #Channel last
  thio_samples = np.zeros((len(all_files),bf_data.shape[0],bf_data.shape[1],1))
  for i in range(len(all_files)):
    thio_samples[i,:,:,0] = np.float32(nibabel.load(all_files[i]).get_fdata())

  #Zero-centered normalization--------------------------------------------------
  thio_samples = (thio_samples - np.mean(thio_samples)) / np.std(thio_samples)
  bf_samples = (bf_samples - np.mean(bf_samples)) / np.std(bf_samples)
    
  #Patch extraction--------------------------------------------------------------
  mp = 2500

  X = []
  Y = []

  for i in range(bf_samples.shape[0]):
    X.append(extract_patches_2d(bf_samples[0,:,:,0],(patch_size,patch_size),max_patches = mp))
    Y.append(extract_patches_2d(thio_samples[0,:,:,0],(patch_size,patch_size),max_patches = mp))

  pX = np.concatenate(X,axis=0)
  pY = np.concatenate(Y,axis=0)

  pX = np.expand_dims(pX,axis=-1)
  pY = np.expand_dims(pY,axis=-1)

  X_train = pX
  Y_train = pY

  joblib.dump(pX,home+'/Exp/BlockFace_'+str(patch_size)+'_training.pkl', compress=True)
  joblib.dump(pY,home+'/Exp/Thio_'+str(patch_size)+'_training.pkl', compress=True)
