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
import monai

import glob
import multiprocessing

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO training')
  parser.add_argument('-d', '--data', help='Input dataset', type=str, required=False, default = 'dhcp')
  parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 50)
  parser.add_argument('-c', '--channels', help='Number of channels in Unet', type=int, required=False, default=32)
  parser.add_argument('-l', '--levels', help='Number of levels in Unet (up to 5)', type=int, required=False, default=3)
  parser.add_argument('-m', '--model', help='Pytorch initialization model', type=str, required=False)

  parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 0)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 4)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 128)
  parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
  parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 128)

  args = parser.parse_args()
  
  max_subjects = 400
  training_split_ratio = 0.95 # use 90% of samples for training, 10% for testing
  num_epochs = args.epochs
  num_workers = args.workers#multiprocessing.cpu_count()
  
  training_batch_size = args.batch_size
  validation_batch_size = args.batch_size
  
  patch_size = args.patch_size
  samples_per_volume = args.samples
  max_queue_length = args.queue
  
  n_channels = args.channels
  n_levels = args.levels
  data = args.data

  prefix = 'unet3d_monai_'
  prefix += data
  prefix += '_epochs_'+str(num_epochs)
  prefix += '_patches_'+str(patch_size)
  prefix += '_sampling_'+str(samples_per_volume)
  prefix += '_nchannels_'+str(n_channels)
  prefix += '_nlevels_'+str(n_levels)

  if args.model is not None:
    prefix += '_using_init'
    
  output_path = home+'/Sync/Experiments/'
  subjects = []

  if data == 'dhcp':
    data_path = home+'/Sync/Data/DHCP/'
    out_channels = 10
    all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
    all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)

    all_seg = all_seg[:max_subjects] 

    for seg_file in all_seg:
      id_subject = seg_file.split('/')[6].split('_')[0:2]
      id_subject = id_subject[0]+'_'+id_subject[1]

      t2_file = [s for s in all_t2s if id_subject in s][0]
        
      subject = tio.Subject(
        image=tio.ScalarImage(t2_file),
        label=tio.LabelMap(seg_file),
      )
      subjects.append(subject)

  if data=='equinus':
    data_path = home+'/Sync/Data/Equinus_Learning/'
    out_channels = 4
    subject_names = ['E01','E02','E03','E05','E06','E08','E09','E10','E13','T01','T02','T03','T04','T05','T06','T07','T08','T09','T10','T11']
    subjects = []

    for s in subject_names:
        
      subject = tio.Subject(
        image=tio.ScalarImage(data_path+'sub_'+s+'_static_3DT1_flirt.nii.gz'),
        label=tio.LabelMap(data_path+'sub_'+s+'_static_3DT1_labels_flirt.nii.gz'),
      )
      subjects.append(subject)

  dataset = tio.SubjectsDataset(subjects)
  print('Dataset size:', len(dataset), 'subjects')
  prefix += '_subj_'+str(len(dataset))


  #%%
  onehot = tio.OneHot()
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
  bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
  noise = tio.RandomNoise(std=0.1, p=0.25)
  prefix += '_bias_flip_affine_noise'

  if data=='dhcp':
    normalization = tio.ZNormalization(masking_method='label')
    spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)

  if data=='equinus':
    normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    #spatial = tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0, p=0.75)
    spatial = tio.OneOf({
        tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
    p=0.75,
    )  

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

# probabilities = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
# sampler = tio.data.LabelSampler(
#           patch_size=patch_size,
#           label_name='label',
#           label_probabilities=probabilities,
# )

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
  
  channels_list = [n_channels]
  if n_levels > 1:
    channels_list.append(n_channels*2)
  if n_levels > 2:
    channels_list.append(n_channels*4)
  if n_levels > 3:
    channels_list.append(n_channels*8)
  if n_levels > 4:
    channels_list.append(n_channels*16)

  channels_tuple = tuple(channels_list)
  unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=out_channels,
    channels=channels_tuple,
    strides=(2, 2, 2),
    num_res_units=2,
  )    
  if args.model is not None:
    unet.load_state_dict(torch.load(args.model))

#%%
  class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
      super().__init__()
      self.lr = learning_rate
      self.net = net
      self.criterion = criterion
      self.optimizer_class = optimizer_class
    
    def forward(self,x):
      return self.net(x)

    def configure_optimizers(self):
      optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
      return optimizer
    
    def prepare_batch(self, batch):
      return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
      x, y = self.prepare_batch(batch)
      y_hat = self.net(x)
      return y_hat, y

    def training_step(self, batch, batch_idx):
      y_hat, y = self.infer_batch(batch)
      loss = self.criterion(y_hat, y)
      self.log('train_loss', loss, prog_bar=True)
      return loss
    
    def validation_step(self, batch, batch_idx):
      y_hat, y = self.infer_batch(batch)
      loss = self.criterion(y_hat, y)
      self.log('val_loss', loss)
      return loss

  net = Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True),
    #criterion=monai.losses.GeneralizedDiceLoss(softmax=True),
    #criterion=monai.losses.TverskyLoss(softmax=True),        
    learning_rate=1e-4,
    optimizer_class=torch.optim.AdamW,
  )

  trainer = pl.Trainer(
    gpus=1,
    max_epochs=num_epochs,
    progress_bar_refresh_rate=20,
    precision=16,
  )
  trainer.logger._default_hp_metric = False

  trainer.fit(net, training_loader_patches, validation_loader_patches)
  trainer.save_checkpoint(output_path+prefix+'.ckpt')
  torch.save(unet.state_dict(), output_path+prefix+'_torch.pt')

  print('Finished Training')        