#!/usr/bin/env python3
#%%
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 14:41:16 2022

@author: rousseau
"""
import os
from os.path import expanduser
home = expanduser("~")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from skimage.transform import resize

#%%
file_npz = home+'/Sync-Exp/Data/dhcp_2d.npz'
if os.path.exists(file_npz):
  array_npz = np.load(file_npz)  
  X = array_npz['arr_0']
  Y = array_npz['arr_1']
  
else:    
  #data_path = '/Volumes/2big/Sync-Data/Cerveau/Neonatal/dhcp-rel-3/'
  data_path = '/media/rousseau/Seagate5To/Sync-Data/Cerveau/Neonatal/dhcp-rel-3/'
    
  file_tsv = pd.read_csv(data_path+'combined.tsv', sep='\t')
  print(file_tsv.head())
  
  age = np.array(file_tsv['scan_age'])
  plt.figure()
  plt.hist(age, bins=25)
  plt.show()
  
  n_images = file_tsv.shape[0]
  
  slices = []
  ages = []
  
  for i in range(n_images):
    subject = str(file_tsv['participant_id'][i])
    session = str(file_tsv['session_id'][i])
    age = file_tsv['scan_age'][i]
    
    
    file_nifti = data_path+'sub-'+subject+'/ses-'+session+'/anat/sub-'+subject+'_ses-'+session+'_desc-restore_T2w.nii.gz'
  
    if os.path.exists(file_nifti):
  
      print(file_nifti)
      im_nifti = nibabel.load(file_nifti) #size : 217 x 290 x 290
      array = im_nifti.get_fdata()
      #print(array.shape)
    
      array_2d = array[:,:,int(array.shape[2]/2)]
      resized_array = resize(array_2d, (192,256))
      
      normalized_array = (resized_array - np.mean(resized_array)) / np.std(resized_array)
      slices.append(np.expand_dims(normalized_array, axis = 0))
      ages.append(age)
        
  X = np.concatenate(slices,axis=0)
  Y = np.array(ages)    
  
  plt.figure()
  plt.imshow(X[0,:,:],cmap='gray')
  plt.show()
  
  np.savez(file_npz,X,Y)

#%%

  