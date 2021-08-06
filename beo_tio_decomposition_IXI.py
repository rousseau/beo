#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

from beo_pl_nets import DecompNet_IXI

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO Decomposition IXI')
  parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 50)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) initialization model', type=str, required=False)
  parser.add_argument('-l', '--latent_dim', help='Dimension of the latent space', type=int, required=False, default = 10)
  parser.add_argument('-f', '--n_filters', help='Number of filters', type=int, required=False, default = 16)
  parser.add_argument('--n_features', help='Number of features', type=int, required=False, default = 16)
  parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 32)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 4)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 64)
  parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
  parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 1024)
  parser.add_argument('--max_subjects', help='Max number of subjects', type=int, required=False, default = 100)
  parser.add_argument('--learning_rate', help='Learning Rate (for optimization)', type=float, required=False, default = 1e-4)

  args = parser.parse_args()
  max_subjects = args.max_subjects
  training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
  num_workers = args.workers
  patch_size = args.patch_size
  patch_overlap = int(patch_size / 2)  

  max_queue_length = args.queue
  samples_per_volume = args.samples
  training_batch_size = args.batch_size
  validation_batch_size = args.batch_size

  latent_dim = args.latent_dim 
  n_filters = args.n_filters
  n_features = args.n_features
  num_epochs = args.epochs

  learning_rate = args.learning_rate

  data_path = home+'/Data/IXI/'
  output_path = home+'/Sync-Exp/Experiments/'

  prefix = 'gromovIXI'
  prefix += '_epochs_'+str(num_epochs)
  prefix += '_subj_'+str(max_subjects)
  prefix += '_patches_'+str(patch_size)
  prefix += '_sampling_'+str(samples_per_volume)
  prefix += '_latentdim_'+str(latent_dim)
  prefix += '_nfilters_'+str(n_filters)
  prefix += '_nfeatures_'+str(n_features)

  all_pds = glob.glob(data_path+'*PD.nii.gz', recursive=True)

  #Scanner info per sequence : TR, TE, FA
  ScannerInfoT1 = {}
  ScannerInfoT1['Guys'] = [9.813, 4.603, 8.0]
  ScannerInfoT1['HH'] = [9.6, 4.60, 8.0]

  ScannerInfoT2 = {}
  ScannerInfoT2['Guys'] = [8178.34, 100.0, 90.0]
  ScannerInfoT2['HH'] = [5725.79, 100.0, 90.0]

  ScannerInfoPD = {}
  ScannerInfoPD['Guys'] = [8178.34, 8.0, 90.0]
  ScannerInfoPD['HH'] = [5725.79, 8.0, 90.0]

  subjects = []
  for pd_file in all_pds:
      id_subject = pd_file.split('/')[5].split('-')[0]
      site = pd_file.split('/')[5].split('-')[1]
      number = pd_file.split('/')[5].split('-')[2]

      t1_file = data_path+id_subject+'-'+site+'-'+number+'-T1.nii.gz'
      t2_file = data_path+id_subject+'-'+site+'-'+number+'-T2.nii.gz'
     
      #if site != 'IOP':
      if site == 'HH':
        subject = tio.Subject(
            pd=tio.ScalarImage(pd_file),
            t1=tio.ScalarImage(t1_file),
            t2=tio.ScalarImage(t2_file),
            zt1=ScannerInfoT1[site],
            zt2=ScannerInfoT2[site],
            zpd=ScannerInfoPD[site],           
        )
        subjects.append(subject)

  subjects = subjects[:max_subjects]
  dataset = tio.SubjectsDataset(subjects)
  print('Dataset size:', len(dataset), 'subjects')

  #%%
  normalization = tio.ZNormalization()

  spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)

  bias = tio.RandomBiasField(coefficients = 0.5, p=0.3)
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
  noise = tio.RandomNoise(std=0.1, p=0.25)

  transforms = [flip, spatial, bias, normalization, noise]

  training_transform = tio.Compose(transforms)
  validation_transform = tio.Compose([normalization])  

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

  #print('Memory required for queue:')
  #print(patches_training_set.get_max_memory_pretty())

  training_loader_patches = torch.utils.data.DataLoader(
      patches_training_set, batch_size=training_batch_size)

  validation_loader_patches = torch.utils.data.DataLoader(
      patches_validation_set, batch_size=validation_batch_size)

  if args.model is not None:
    net = DecompNet_IXI().load_from_checkpoint(args.model, latent_dim = latent_dim, n_filters = n_filters, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)
  else:
    net = DecompNet_IXI(latent_dim = latent_dim, n_filters = n_filters, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)

  checkpoint_callback = ModelCheckpoint(
    dirpath=output_path,
    filename=prefix+'_{epoch:02d}',
    verbose=True
    )

  trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])

  #%%

  trainer.fit(net, training_loader_patches, validation_loader_patches)
  trainer.save_checkpoint(output_path+prefix+'.ckpt')

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
  aggregator_rx = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_ry = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_rz = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_cx = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_cy = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_cz = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_fx = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_fy = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_fz = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
  aggregator_my2x = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

  net.eval()

  with torch.no_grad():
    for patches_batch in patch_loader:
      x = patches_batch['t1'][tio.DATA]
      y = patches_batch['t2'][tio.DATA]
      z = patches_batch['pd'][tio.DATA]
      locations = patches_batch[tio.LOCATION]

      rx, ry, rz, cx, cy, cz, fx, fy, fz, zx, zy, zz, my2x, mx2y = net(x,y,z)
      aggregator_rx.add_batch(rx, locations)
      aggregator_ry.add_batch(ry, locations)
      aggregator_rz.add_batch(rz, locations)
      aggregator_cx.add_batch(cx, locations)
      aggregator_cy.add_batch(cy, locations)
      aggregator_cz.add_batch(cz, locations)
      aggregator_fx.add_batch(fx, locations)
      aggregator_fy.add_batch(fy, locations)
      aggregator_fz.add_batch(fz, locations)
      aggregator_my2x.add_batch(my2x, locations)

  output_rx = aggregator_rx.get_output_tensor()
  output_ry = aggregator_ry.get_output_tensor()
  output_rz = aggregator_rz.get_output_tensor()
  output_cx = aggregator_cx.get_output_tensor()
  output_cy = aggregator_cy.get_output_tensor()
  output_cz = aggregator_cz.get_output_tensor()
  output_fx = aggregator_fx.get_output_tensor()
  output_fy = aggregator_fy.get_output_tensor()
  output_fz = aggregator_fz.get_output_tensor()
  output_my2x = aggregator_my2x.get_output_tensor()
  
  print('Saving images...')

  o_rx = tio.ScalarImage(tensor=output_rx, affine=subject['t1'].affine)
  o_rx.save(output_path+'gromov_rx.nii.gz')
  o_ry = tio.ScalarImage(tensor=output_ry, affine=subject['t1'].affine)
  o_ry.save(output_path+'gromov_ry.nii.gz')
  o_rz = tio.ScalarImage(tensor=output_rz, affine=subject['t1'].affine)
  o_rz.save(output_path+'gromov_rz.nii.gz')
  o_cx = tio.ScalarImage(tensor=output_cx, affine=subject['t1'].affine)
  o_cx.save(output_path+'gromov_cx.nii.gz')
  o_cy = tio.ScalarImage(tensor=output_cy, affine=subject['t1'].affine)
  o_cy.save(output_path+'gromov_cy.nii.gz')
  o_cz = tio.ScalarImage(tensor=output_cz, affine=subject['t1'].affine)
  o_cz.save(output_path+'gromov_cz.nii.gz')
  #o_fx = tio.ScalarImage(tensor=output_fx, affine=subject['t1'].affine)
  #o_fx.save(output_path+'gromov_fx.nii.gz')
  #o_fy = tio.ScalarImage(tensor=output_fy, affine=subject['t1'].affine)
  #o_fy.save(output_path+'gromov_fy.nii.gz')
  output_f = torch.mean(torch.stack([output_fx,output_fy,output_fz]),dim=0)
  o_f = tio.ScalarImage(tensor=output_f, affine=subject['t1'].affine)
  o_f.save(output_path+'gromov_f.nii.gz')

  o_my2x = tio.ScalarImage(tensor=output_my2x, affine=subject['t1'].affine)
  o_my2x.save(output_path+'gromov_my2x.nii.gz')


  subject['t2'].save(output_path+'gromov_t2.nii.gz')
  subject['t1'].save(output_path+'gromov_t1.nii.gz')
  subject['pd'].save(output_path+'gromov_pd.nii.gz')
