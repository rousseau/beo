#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import numpy as np
import nibabel
import SimpleITK as sitk

def load_images(data_path, key, loader = 'itk', verbose = 0):  
  """Load image data in a specified directory using a key word"""  
  """Data can be loaded using itk or nibabel"""
  if verbose !=0:
    print('loading images')
  images = []
  #load all the images
  directories = glob.glob(data_path, recursive=True)
  if verbose !=0:
    print(directories)
  all_files = []
  for d in directories:
    if verbose !=0:
      print('Looking for data in '+d)
    files = glob.glob(d+key)
    all_files.extend(files)
  all_files.sort()
  
  for file in all_files:  
    if verbose !=0:
      print('Loading : '+file)
    if loader == 'itk':
      #Load images using SimpleITK
      images.append(sitk.ReadImage(file))  
    else:      
      #Load images using nibabel 
      images.append(nibabel.load(file))
    
  return images