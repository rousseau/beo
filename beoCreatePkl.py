#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
from os.path import expanduser
home = expanduser("~")
from dataset import get_ixi_2dpatches, get_hcp_2dpatches, get_hcp_4darrays

patch_size = 128
dataset = 'HCP'
ratio_training = 0.75

if dataset == 'IXI':  
  n_patches = 100 #per slice 
  (T1_2D,T2_2D,PD_2D) = get_ixi_2dpatches(patch_size = patch_size, n_patches = n_patches)
  joblib.dump(T1_2D,home+'/Exp/T1_2D'+str(patch_size)+'.pkl', compress=True)
  joblib.dump(T2_2D,home+'/Exp/T2_2D'+str(patch_size)+'.pkl', compress=True)
  joblib.dump(PD_2D,home+'/Exp/PD_2D'+str(patch_size)+'.pkl', compress=True)

if dataset == 'HCP':  
  n_patches = 25 #per slice 
  
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
