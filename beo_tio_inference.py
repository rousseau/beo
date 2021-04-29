#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:43:26 2021

@author: rousseau
"""
from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from beo_pl_nets import Unet

import glob

max_subjects = 50
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing

data_path = home+'/Sync/Data/DHCP/'
output_path = home+'/Sync/Experiments/'

all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)

all_seg = all_seg[:max_subjects]
subjects = []

for seg_file in all_seg:
    id_subject = seg_file.split('/')[6].split('_')[0:2]
    id_subject = id_subject[0]+'_'+id_subject[1]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    
    subject = tio.Subject(
        t2=tio.ScalarImage(t2_file),
        seg=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')

training_transform = tio.Compose([
  #tio.ToCanonical(),
  #tio.RandomMotion(p=0.2),
  #tio.RandomBiasField(p=0.3),
  tio.ZNormalization(masking_method=tio.ZNormalization.mean),
  #tio.RandomNoise(p=0.5),
  #tio.RandomFlip(),
  #tio.OneOf({
  #  tio.RandomAffine(): 0.8,
  #  tio.RandomElasticDeformation(): 0.2,
  #}),
  tio.OneHot(),
])

validation_transform = tio.Compose([
  #tio.ToCanonical(),
  tio.ZNormalization(masking_method=tio.ZNormalization.mean),
  tio.OneHot(),
])

num_subjects = len(dataset)
num_training_subjects = int(training_split_ratio * num_subjects)
num_validation_subjects = num_subjects - num_training_subjects

num_split_subjects = num_training_subjects, num_validation_subjects
training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

print('Training set:', len(training_set), 'subjects')
print('Validation set:', len(validation_set), 'subjects')



print('Inference')

patch_overlap = 32
patch_size = 64, 64, 64
batch_size = 16

subject = validation_set[0]

grid_sampler = tio.inference.GridSampler(
  subject,
  patch_size,
  patch_overlap,
  )

patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

net = Unet.load_from_checkpoint(output_path+'example_model.ckpt')
net.eval()

with torch.no_grad():
  for patches_batch in patch_loader:
    input_tensor = patches_batch['t2'][tio.DATA]
    locations = patches_batch[tio.LOCATION]
    outputs = net(input_tensor)
    aggregator.add_batch(outputs, locations)
output_tensor = aggregator.get_output_tensor()

print(output_tensor.shape)

print('Saving images')
output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['t2'].affine)
output_seg.save(output_path+'toto.nii.gz')
subject['t2'].save(output_path+'toto_t2.nii.gz')
subject['seg'].save(output_path+'toto_seg.nii.gz')

print(output_seg.affine)
print(subject['t2'].affine)
print(subject['seg'].affine)