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

from beo_pl_nets import DecompNet_IXI, DecompNet_3, DecompNet_VAE, MTL, MTL_net, MTL_loss

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO Decomposition IXI')
  parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 50)
  parser.add_argument('-m', '--model', help='Pytorch lightning (ckpt file) initialization model', type=str, required=False)
  parser.add_argument('-l', '--latent_dim', help='Dimension of the latent space', type=int, required=False, default = 10)
  #parser.add_argument('-f', '--n_filters', help='Number of filters', type=int, required=False, default = 16)
  parser.add_argument('--n_filters_encoder', help='Number of filters for latent space encoding', type=int, required=False, default = 8)
  parser.add_argument('--n_filters_feature', help='Number of filters for unet-based feature estimation', type=int, required=False, default = 16)
  parser.add_argument('--n_filters_recon', help='Number of filters for reconstruction', type=int, required=False, default = 16)
  parser.add_argument('--n_features', help='Number of features', type=int, required=False, default = 16)
  parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 32)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 64)
  parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
  parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 1024)
  parser.add_argument('--max_subjects', help='Max number of subjects', type=int, required=False, default = 100)
  parser.add_argument('--learning_rate', help='Learning Rate (for optimization)', type=float, required=False, default = 1e-4)
  parser.add_argument('--prefix', help='Prefix of the output model name', type=str, required=False, default = 'gromovIXI')
  parser.add_argument('--loss_recon', help='Reconstruction Loss', type=float, required=False, default = 1)
  parser.add_argument('--loss_cross', help='Cross-modality reconstruction Loss', type=float, required=False, default = 1)
  parser.add_argument('--loss_feat', help='Feature similarity Loss', type=float, required=False, default = 1)
  parser.add_argument('--loss_kld', help='Kullback Liebler Divergence Loss', type=float, required=False, default = 0.1)
  parser.add_argument('--loss_mapping', help='Mapping Loss', type=float, required=False, default = 0)

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
  #n_filters = args.n_filters
  n_filters_encoder = args.n_filters_encoder
  n_filters_feature = args.n_filters_feature
  n_filters_recon = args.n_filters_recon
  n_features = args.n_features
  num_epochs = args.epochs

  learning_rate = args.learning_rate

  loss_dict = {}
  loss_dict['recon'] = args.loss_recon
  loss_dict['cross'] = args.loss_cross  
  loss_dict['feat'] = args.loss_feat
  loss_dict['kld'] = args.loss_kld
  loss_dict['mapping'] = args.loss_mapping

  data_path = home+'/Data/IXI/'
  output_path = home+'/Sync-Exp/Experiments/'

  prefix = args.prefix
  prefix += '_loss_'+str(loss_dict['recon'])+'_'+str(loss_dict['cross'])+'_'+str(loss_dict['feat'])+'_'+str(loss_dict['kld'])+'_'+str(loss_dict['mapping'])
  prefix += '_epochs_'+str(num_epochs)
  prefix += '_subj_'+str(max_subjects)
  prefix += '_patches_'+str(patch_size)
  prefix += '_sampling_'+str(samples_per_volume)
  prefix += '_latentdim_'+str(latent_dim)
  prefix += '_nfe_'+str(n_filters_encoder)
  prefix += '_nff_'+str(n_filters_feature)
  prefix += '_nfr_'+str(n_filters_recon)
  prefix += '_nfeatures_'+str(n_features)

  all_pds = glob.glob(data_path+'*PD-N4.nii.gz', recursive=True)

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

      t1_file = data_path+id_subject+'-'+site+'-'+number+'-T1-N4.nii.gz'
      t2_file = data_path+id_subject+'-'+site+'-'+number+'-T2-N4.nii.gz'
     
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

  transforms = [flip, spatial, normalization]

  training_transform = tio.Compose(transforms)
  validation_transform = tio.Compose([normalization])  

  #%%
  seed = 42  # for reproducibility

  num_subjects = len(dataset)
  num_training_subjects = int(training_split_ratio * num_subjects)
  num_validation_subjects = num_subjects - num_training_subjects

  num_split_subjects = num_training_subjects, num_validation_subjects
  training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects, generator=torch.Generator().manual_seed(seed))

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

  # if args.model is not None:
  #   net = DecompNet_VAE().load_from_checkpoint(args.model, latent_dim = latent_dim, n_filters_encoder = n_filters_encoder, n_filters_feature = n_filters_feature, n_filters_recon = n_filters_recon, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)
  # else:
  #   net = DecompNet_VAE(latent_dim = latent_dim, n_filters_encoder = n_filters_encoder, n_filters_feature = n_filters_feature, n_filters_recon = n_filters_recon, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)

  # net.set_loss(loss_dict)

  if args.model is not None:
    net = MTL().load_from_checkpoint(args.model, latent_dim = latent_dim, n_filters_encoder = n_filters_encoder, n_filters_feature = n_filters_feature, n_filters_recon = n_filters_recon, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)
  else:
    net = MTL(latent_dim = latent_dim, n_filters_encoder = n_filters_encoder, n_filters_feature = n_filters_feature, n_filters_recon = n_filters_recon, n_features = n_features, patch_size = patch_size, learning_rate = learning_rate)
  net.net_model.set_dict(loss_dict)
  net.loss_model.set_dict(loss_dict)

  print('losses:')
  for k in loss_dict.keys():
    print(k, ':', loss_dict[k])

  checkpoint_callback = ModelCheckpoint(
    dirpath=output_path,
    filename=prefix+'_{epoch:02d}', 
    verbose=True
    )

  trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20, callbacks=[checkpoint_callback], auto_lr_find=True)
  #trainer.tune(net, training_loader_patches, validation_loader_patches)
  #%%
  if num_epochs > 0:
    trainer.fit(net, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(output_path+prefix+'.ckpt')

  print('Finished Training')
  print(list(net.loss_model.parameters()))
  
  #%%
  print('Inference')

  subject = validation_set[0]

  grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
    )

  patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

  output_keys = ['rx','ry','rz','cx','cy','cz','fx','fy','fz','zxf','zyf','zzf']

  aggregators = {}
  for k in output_keys:
    aggregators[k] = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

  net.eval()

  with torch.no_grad():
    for patches_batch in patch_loader:
      x = patches_batch['t1'][tio.DATA]
      y = patches_batch['t2'][tio.DATA]
      z = patches_batch['pd'][tio.DATA]
      locations = patches_batch[tio.LOCATION]

      output_dict = net.net_model(x,y,z)

      for k in output_keys:
        aggregators[k].add_batch(output_dict[k], locations)  

  print('Saving images...')
  for k in output_keys:
    output = aggregators[k].get_output_tensor()
    o = tio.ScalarImage(tensor=output, affine=subject['t1'].affine)
    o.save(output_path+'gromov_'+k+'.nii.gz')

  #output_f = torch.mean(torch.stack([output_fx,output_fy,output_fz]),dim=0)
  #o_f = tio.ScalarImage(tensor=output_f, affine=subject['t1'].affine)
  #o_f.save(output_path+'gromov_f.nii.gz')

  subject['t2'].save(output_path+'gromov_t2.nii.gz')
  subject['t1'].save(output_path+'gromov_t1.nii.gz')
  subject['pd'].save(output_path+'gromov_pd.nii.gz')
