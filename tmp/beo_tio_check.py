#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:05:23 2021

@author: rousseau
"""


from os.path import expanduser
home = expanduser("~")
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchio as tio

image_file = home+'/Sync-Exp/Data/DHCP/sub-CC00060XX03_ses-12501_desc-restore_T2w.nii.gz'
label_file = home+'/Sync-Exp/Data/DHCP/sub-CC00060XX03_ses-12501_desc-fusion_space-T2w_dseg.nii.gz'


subject = tio.Subject(
  image=tio.ScalarImage(image_file),
  label=tio.LabelMap(label_file),
)

subject = tio.Subject(
  hr=tio.ScalarImage(image_file),
  lr=tio.ScalarImage(image_file),
)

#normalization = tio.ZNormalization(masking_method='label')#masking_method=tio.ZNormalization.mean)
normalization = tio.ZNormalization()
onehot = tio.OneHot()

spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0)

bias = tio.RandomBiasField(coefficients = 0.5, p=0) 
flip = tio.RandomFlip(axes=('LR',), p=1)
noise = tio.RandomNoise(std=0.1, p=0)

sampling_1mm = tio.Resample(1)
sampling_05mm = tio.Resample(0.5)
blur = tio.RandomBlur(0.5)

sampling_jess = tio.Resample((0.8,0.8,2), exclude='hr')
blur_jess = tio.Blur(std=(0.001,0.001,1), exclude='hr')
downsampling_jess = tio.Resample((0.8,0.8,2), exclude='hr')
upsampling_jess = tio.Resample(target='hr', exclude='hr')

tocanonical = tio.ToCanonical()
crop1 = tio.CropOrPad((290,290,200))
crop2 = tio.CropOrPad((290,290,200), include='lr')

#transforms = tio.Compose([spatial, bias, flip, normalization, noise])
#transforms = tio.Compose([normalization, sampling_1mm, noise, blur, sampling_05mm])
#transforms = tio.Compose([blur_jess,sampling_jess])

transforms = tio.Compose([tocanonical, crop1, flip, spatial, normalization, blur_jess, downsampling_jess, upsampling_jess])

transformed_subject = transforms(subject)
#transformed_subject.plot()

print(transformed_subject.get_composed_history())

print(transformed_subject['hr'].affine)
print(transformed_subject['hr'].origin)
print(transformed_subject['hr'].spatial_shape)
print(transformed_subject['lr'].affine)
print(transformed_subject['lr'].origin)
print(transformed_subject['lr'].spatial_shape)

#transformed_subject.image.save(home+'/Sync/Data/MarsFet/tmp.nii.gz')

#transformed_subject.image.save(home+'/Sync-Exp/tmp.nii.gz')


#%%
# Transforms for 3 images
subject = tio.Subject(
  hr=tio.ScalarImage(image_file),
  lr_1=tio.ScalarImage(image_file),
  lr_2=tio.ScalarImage(image_file),
  lr_3=tio.ScalarImage(image_file))
 
normalization = tio.ZNormalization()
spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)

tocanonical = tio.ToCanonical()

b1 = tio.Blur(std=(0.001,0.001,1), include='lr_1') #blur
d1 = tio.Resample((0.8,0.8,2), include='lr_1')     #downsampling
u1 = tio.Resample(target='hr', include='lr_1')     #upsampling

b2 = tio.Blur(std=(0.001,1,0.001), include='lr_2') #blur
d2 = tio.Resample((0.8,2,0.8), include='lr_2')     #downsampling
u2 = tio.Resample(target='hr', include='lr_2')     #upsampling

b3 = tio.Blur(std=(1,0.001,0.001), include='lr_3') #blur
d3 = tio.Resample((2,0.8,0.8), include='lr_3')     #downsampling
u3 = tio.Resample(target='hr', include='lr_3')     #upsampling

#transforms = tio.Compose([tocanonical, flip, spatial, normalization, b1, d1, u1, b2, d2, u2, b3, d3, u3])
transforms = tio.Compose([tocanonical, d1, u1, d2, u2, d3, u3])

print(transformed_subject.get_composed_history())

transformed_subject = transforms(subject)
transformed_subject.plot()
