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
from pytorch_lightning.callbacks import ModelCheckpoint

import glob
import multiprocessing

from beo_pl_nets import DecompNet

max_subjects = 10
training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
num_epochs = 5
num_workers = 0#multiprocessing.cpu_count()

training_batch_size = 1
validation_batch_size = 1 

patch_size = 64
samples_per_volume = 32
max_queue_length = 256

prefix = 'gromov'
prefix += '_epochs_'+str(num_epochs)
prefix += '_subj_'+str(max_subjects)
prefix += '_patches_'+str(patch_size)
prefix += '_sampling_'+str(samples_per_volume)

data_path = home+'/Sync/Data/DHCP/'
output_path = home+'/Sync/Experiments/'

all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

all_t1s = all_t1s[:max_subjects]
subjects = []

for t1_file in all_t1s:
    id_subject = t1_file.split('/')[6].split('_')[0:2]
    id_subject = id_subject[0]+'_'+id_subject[1]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    seg_file = [s for s in all_seg if id_subject in s][0]
    
    subject = tio.Subject(
        t1=tio.ScalarImage(t1_file),
        t2=tio.ScalarImage(t2_file),
        seg=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)
print('Dataset size:', len(dataset), 'subjects')

#%%
normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
onehot = tio.OneHot()

spatial = tio.OneOf({
    tio.RandomAffine(scales=0.1,degrees=30): 0.8,
    #tio.RandomElasticDeformation(): 0.2,
  },
  p=0.75,
)

bias = tio.RandomBiasField(p=0.3)
flip = tio.RandomFlip()
noise = tio.RandomNoise(p=0.5)

transforms = [bias, normalization, flip, spatial, noise, onehot]

training_transform = tio.Compose(transforms)
validation_transform = tio.Compose([normalization, onehot])

#%%
seed = 42  # for reproducibility


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

#%%

print('num_workers : '+str(num_workers))

probabilities = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
#sampler = tio.data.UniformSampler(patch_size)
sampler = tio.data.LabelSampler(
          patch_size=patch_size,
          label_name='seg',
          label_probabilities=probabilities,
)

patches_training_set = tio.Queue(
    subjects_dataset=training_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=True,
    shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_set,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=sampler,
    num_workers=num_workers,
    shuffle_subjects=False,
    shuffle_patches=False,
)

training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

validation_loader_patches = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)

#%%

net = DecompNet()

checkpoint_callback = ModelCheckpoint(
  dirpath=output_path,
  filename=prefix+'_{epoch:02d}',
  verbose=True
  )

trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])

#%%

trainer.fit(net, training_loader_patches, validation_loader_patches)

print('Finished Training')