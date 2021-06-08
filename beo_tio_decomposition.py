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

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO Decomposition')
  parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 50)
  parser.add_argument('-l', '--latent_dim', help='Dimension of the latent space', type=int, required=False, default = 10)
  parser.add_argument('-f', '--n_filters', help='Number of filters', type=int, required=False, default = 16)
  parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 0)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 4)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 128)
  parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
  parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 32)

  args = parser.parse_args()

  max_subjects = 200
  training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
  num_epochs = args.epochs
  num_workers = args.workers#multiprocessing.cpu_count()


  training_batch_size = args.batch_size
  validation_batch_size = args.batch_size
  patch_size = args.patch_size
  patch_overlap = int(patch_size / 2)  

  max_queue_length = args.queue

  samples_per_volume = args.samples

  latent_dim = args.latent_dim 
  n_filters = args.n_filters
  prefix = 'gromov'
  prefix += '_epochs_'+str(num_epochs)
  prefix += '_subj_'+str(max_subjects)
  prefix += '_patches_'+str(patch_size)
  prefix += '_sampling_'+str(samples_per_volume)
  prefix += '_latentdim_'+str(latent_dim)
  prefix += '_nfilters_'+str(n_filters)

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
          label=tio.LabelMap(seg_file),
      )
      subjects.append(subject)

  dataset = tio.SubjectsDataset(subjects)
  print('Dataset size:', len(dataset), 'subjects')

  #%%
  normalization = tio.ZNormalization(masking_method='label')
  onehot = tio.OneHot()

  spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)

  bias = tio.RandomBiasField(coefficients = 0.5, p=0.3)
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
  noise = tio.RandomNoise(std=0.1, p=0.25)

  transforms = [flip, spatial, bias, normalization, noise, onehot]

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

  sampler = tio.data.UniformSampler(patch_size)
  #probabilities = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
  #sampler = tio.data.LabelSampler(
  #          patch_size=patch_size,
  #          label_name='label',
  #          label_probabilities=probabilities,
  #)

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

  #net = DecompNet(latent_dim = latent_dim, n_filters = n_filters, patch_size = patch_size)
  net = DecompNet().load_from_checkpoint('gromov_epochs_40_subj_100_patches_128_sampling_8_latentdim_10_nfilters_16.ckpt', latent_dim = latent_dim, n_filters = n_filters, patch_size = patch_size)

  checkpoint_callback = ModelCheckpoint(
    dirpath=output_path,
    filename=prefix+'_{epoch:02d}',
    verbose=True
    )

  trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])

  #%%

  trainer.fit(net, training_loader_patches, validation_loader_patches)
  trainer.save_checkpoint(output_path+prefix+'.ckpt')
  #net = DecompNet.load_from_checkpoint(checkpoint_path=output_path+prefix+'.ckpt')

  print('Finished Training')

  #%%
  print('Inference')

  subject = validation_set[0]

  grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
    )

  patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
  aggregator_xhat = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_yhat = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_rx = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_ry = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_fx = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_fy = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_sx = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_sy = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  net.eval()

  with torch.no_grad():
    for patches_batch in patch_loader:
      x_tensor = patches_batch['t1'][tio.DATA]
      y_tensor = patches_batch['t2'][tio.DATA]
      locations = patches_batch[tio.LOCATION]
      x_hat, y_hat, rx, ry, fx, fy, sx, sy = net(x_tensor, y_tensor)
      aggregator_xhat.add_batch(x_hat, locations)
      aggregator_yhat.add_batch(y_hat, locations)
      aggregator_rx.add_batch(rx, locations)
      aggregator_ry.add_batch(ry, locations)
      aggregator_fx.add_batch(fx, locations)
      aggregator_fy.add_batch(fy, locations)
      aggregator_sx.add_batch(sx, locations)
      aggregator_sy.add_batch(sy, locations)

  output_xhat = aggregator_xhat.get_output_tensor()
  output_yhat = aggregator_yhat.get_output_tensor()
  output_rx = aggregator_rx.get_output_tensor()
  output_ry = aggregator_ry.get_output_tensor()
  output_fx = aggregator_fx.get_output_tensor()
  output_fy = aggregator_fy.get_output_tensor()
  output_sx = aggregator_sx.get_output_tensor()
  output_sy = aggregator_sy.get_output_tensor() 

  print('Saving images...')

  o_xhat = tio.ScalarImage(tensor=output_xhat, affine=subject['t1'].affine)
  o_xhat.save(output_path+'gromov_xhat.nii.gz')
  o_yhat = tio.ScalarImage(tensor=output_yhat, affine=subject['t1'].affine)
  o_yhat.save(output_path+'gromov_yhat.nii.gz')
  o_rx = tio.ScalarImage(tensor=output_rx, affine=subject['t1'].affine)
  o_rx.save(output_path+'gromov_rx.nii.gz')
  o_ry = tio.ScalarImage(tensor=output_ry, affine=subject['t1'].affine)
  o_ry.save(output_path+'gromov_ry.nii.gz')
  o_fx = tio.ScalarImage(tensor=output_fx, affine=subject['t1'].affine)
  o_fx.save(output_path+'gromov_fx.nii.gz')
  o_fy = tio.ScalarImage(tensor=output_fy, affine=subject['t1'].affine)
  o_fy.save(output_path+'gromov_fy.nii.gz')
  o_sx = tio.ScalarImage(tensor=output_sx, affine=subject['t1'].affine)
  o_sx.save(output_path+'gromov_sx.nii.gz')
  o_sy = tio.ScalarImage(tensor=output_sy, affine=subject['t1'].affine)
  o_sy.save(output_path+'gromov_sy.nii.gz')


  o_sx_bin = torch.unsqueeze(torch.argmax(output_sx,dim=0),0).int()
  o_sx_bin = tio.LabelMap(tensor=o_sx_bin, affine=subject['t1'].affine)
  o_sx_bin.save(output_path+'gromov_sx_labels.nii.gz')

  o_sy_bin = torch.unsqueeze(torch.argmax(output_sy,dim=0),0).int()
  o_sy_bin = tio.LabelMap(tensor=o_sy_bin, affine=subject['t1'].affine)
  o_sy_bin.save(output_path+'gromov_sy_labels.nii.gz')

  subject['t2'].save(output_path+'gromov_t2.nii.gz')
  subject['t1'].save(output_path+'gromov_t1.nii.gz')
  subject['label'].save(output_path+'gromov_seg.nii.gz')
