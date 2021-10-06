#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader, IterableDataset, Dataset
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data.dataset import BufferedShuffleDataset
import random
from itertools import islice

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import glob
import multiprocessing

import argparse


class BatchedPatchesDataset(IterableDataset):
  def __init__(self, subjects_datasets, weights, sampler, samples_per_volume):
    self.subjects_datasets = subjects_datasets
    self.weights = weights
    self.sampler = sampler
    self.samples_per_volume = samples_per_volume

  def __iter__(self):
    while True:
      sampled_dataset = random.choices(population=self.subjects_datasets, weights=self.weights)[0]
      idx = random.randint(0, len(sampled_dataset) - 1)
      sample = sampled_dataset[idx]
      iterable = self.sampler(sample)
      patches = list(islice(iterable, self.samples_per_volume))

      yield patches

class UnbatchDataset(IterableDataset):
  def __init__(
    self,
    dataset: Dataset,
    num_workers: int = 0,
  ):
    self.loader = DataLoader(dataset,
                             batch_size=None,
                             num_workers=num_workers)

  def __iter__(self):
    for batch in self.loader:
      yield from batch

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo Disentagled Learning DHCP')
  parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 50)
  parser.add_argument('--n_features', help='Number of features', type=int, required=False, default = 16)
  parser.add_argument('--n_filters_unet', help='Number of filters in Unet', type=int, required=False, default=32)
  parser.add_argument('-a','--activation_unet', help='Final activation function in the unet (relu, tanh, gumbel, softmax)', type=str, required=False, default = 'relu')
  parser.add_argument('-l', '--levels', help='Number of levels in Unet (up to 5)', type=int, required=False, default=3)
  parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 32)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 1)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 64)
  parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
  parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 1024)
  parser.add_argument('--max_subjects', help='Max number of subjects', type=int, required=False, default = 100)
  parser.add_argument('--learning_rate', help='Learning Rate (for optimization)', type=float, required=False, default = 1e-4)
  parser.add_argument('--prefix', help='Prefix of the output model name', type=str, required=False, default = 'DisDHCP')
  parser.add_argument('-m', '--model', help='Pytorch Lightning model', type=str, required=False)

  args = parser.parse_args()
  max_subjects = args.max_subjects
  training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
  num_workers = args.workers
  patch_size = args.patch_size
  patch_overlap = int(patch_size / 2)  

  n_filters_unet = args.n_filters_unet
  activation_unet = args.activation_unet

  n_features = args.n_features
  num_epochs = args.epochs
  learning_rate = args.learning_rate

  max_queue_length = args.queue
  samples_per_volume = args.samples
  training_batch_size = args.batch_size
  validation_batch_size = args.batch_size

  #data_path = home+'/Sync-Exp/Data/DHCP/'
  data_path = home+'/Data/HCP100_T1T2/'

  output_path = home+'/Sync-Exp/Experiments/'
  subjects = []

  #all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
  #all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
  #all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

  all_seg = glob.glob(data_path+'**/*mask.nii.gz', recursive=True)
  all_t2s = glob.glob(data_path+'**/*_T2.nii.gz', recursive=True)
  all_t1s = glob.glob(data_path+'**/*_T1.nii.gz', recursive=True)

  prefix = args.prefix
  prefix += '_epochs_'+str(num_epochs)
  #prefix += '_subj_'+str(max_subjects)
  #prefix += '_patches_'+str(patch_size)
  #prefix += '_sampling_'+str(samples_per_volume)
  prefix += '_act_'+str(activation_unet)

  all_seg = all_seg[:max_subjects] 

  for seg_file in all_seg:
    #id_subject = seg_file.split('/')[6].split('_')[0:2]
    #id_subject = id_subject[0]+'_'+id_subject[1]
    id_subject = seg_file.split('/')[5].split('_')[0]

    t2_file = [s for s in all_t2s if id_subject in s][0]
    t1_file = [s for s in all_t1s if id_subject in s][0]
      
    subject = tio.Subject(
      t2=tio.ScalarImage(t2_file),
      t1=tio.ScalarImage(t1_file),
      label=tio.LabelMap(seg_file),
    )
    subjects.append(subject)

  dataset = tio.SubjectsDataset(subjects)
  print('Dataset size:', len(dataset), 'subjects')

  onehot = tio.OneHot()
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
  bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
  noise = tio.RandomNoise(std=0.1, p=0.25)
  normalization = tio.ZNormalization(masking_method='label')
  spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)

  transforms = [flip, spatial, bias, normalization, noise, onehot]

  training_transform = tio.Compose(transforms)
  validation_transform = tio.Compose([normalization, onehot])

#%%
  num_subjects = len(dataset)
  num_training_subjects = int(training_split_ratio * num_subjects)
  num_validation_subjects = num_subjects - num_training_subjects

  num_split_subjects = num_training_subjects, num_validation_subjects
  training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects, generator=torch.Generator().manual_seed(42))

  training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

  validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

  print('Training set:', len(training_set), 'subjects')
  print('Validation set:', len(validation_set), 'subjects')

#%%
  print('num_workers : '+str(num_workers))

  probabilities = {0: 0, 1: 1}
  #sampler = tio.data.UniformSampler(patch_size)
  sampler = tio.data.LabelSampler(
          patch_size=patch_size,
          label_name='label',
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


  # This yields a dataset with non random batches of batch_size 'samples_per_volume' 
  to_patches = BatchedPatchesDataset(subjects_datasets=[training_set],
                                     weights=[1], # Only relevant when sampling from multiple subject datasets
                                     sampler=sampler,
                                     samples_per_volume=samples_per_volume)
  # Unbatch the batches
  patches_unbatched = UnbatchDataset(to_patches, num_workers)

  # Shuffle to get the patches in a random order
  queue = BufferedShuffleDataset(patches_unbatched, max_queue_length)

  #training_loader_patches = DataLoader(queue, batch_size=args.batch_size)

  training_loader_patches = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)

  validation_loader_patches = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)


#%%
  class Unet(pl.LightningModule):
    def __init__(self, in_channels = 1, out_channels = 10, n_filters = 32, activation = 'relu'):
      super(Unet, self).__init__()

      self.in_channels = in_channels
      self.out_channels = out_channels
      self.n_features = n_filters
      self.activation = activation

      def double_conv(in_channels, out_channels):
          return nn.Sequential(
              nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm3d(out_channels),
              nn.ReLU(inplace=True),
              nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm3d(out_channels),
              nn.ReLU(inplace=True),
          )

      self.dc1 = double_conv(self.in_channels, self.n_features)
      self.dc2 = double_conv(self.n_features, self.n_features*2)
      self.dc3 = double_conv(self.n_features*2, self.n_features*4)
      self.dc4 = double_conv(self.n_features*6, self.n_features*2)
      self.dc5 = double_conv(self.n_features*3, self.n_features)
      self.mp = nn.MaxPool3d(2)

      self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

      self.out = nn.Conv3d(self.n_features, self.out_channels, kernel_size=1)

    def forward(self, x):
      x1 = self.dc1(x)

      x2 = self.mp(x1)
      x2 = self.dc2(x2)

      x3 = self.mp(x2)
      x3 = self.dc3(x3)

      x4 = self.up(x3)
      x4 = torch.cat([x4,x2], dim=1)
      x4 = self.dc4(x4)

      x5 = self.up(x4)
      x5 = torch.cat([x5,x1], dim=1)
      x5 = self.dc5(x5)

      xout = self.out(x5)

      if self.activation == 'tanh':
        return nn.Tanh()(xout)
      elif self.activation == 'softmax':
        return nn.Softmax(dim=1)(xout)
      elif self.activation == 'gumbel':
        return nn.functional.gumbel_softmax(xout, hard=True)
      else:
        return nn.ReLU()(xout)      

#%%
  class Reconstruction(torch.nn.Module):
    def __init__(self, in_channels, n_filters = 16):
      super(Reconstruction, self).__init__()
      
      ks = 3
      pad = 1
      self.recon = nn.Sequential(
        nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
        nn.ReLU(),
        nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
        nn.ReLU(),
        nn.Conv3d(in_channels = n_filters, out_channels = n_filters, kernel_size = ks,stride = 1, padding=pad),
        nn.ReLU(),
        nn.Conv3d(in_channels = n_filters, out_channels = 1, kernel_size = ks,stride = 1, padding=pad)
        )    

    def forward(self,x):
      return self.recon(x)    

  class Feature2Segmentation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_filters = 16):
      super(Feature2Segmentation, self).__init__()
      
      self.n_filters = n_filters

      self.seg = nn.Sequential(
        nn.Conv3d(in_channels = in_channels, out_channels = n_filters, kernel_size = 3,stride = 1, padding=1),
        nn.ReLU(),
        nn.Conv3d(in_channels = n_filters, out_channels = out_channels, kernel_size = 3,stride = 1, padding=1)
        #nn.Sigmoid()
        )  

    def forward(self,x):
      return self.seg(x)        

#%%
  class Disentangler(pl.LightningModule):

    def __init__(self, n_features = 16, n_filters_unet = 16, activation_unet = 'relu', learning_rate = 1e-4):
      super().__init__()
      self.save_hyperparameters()

      self.n_classes = 2 #DHCP : 10, HCP : ?
      self.learning_rate = learning_rate

      self.unet = Unet(in_channels = 1, out_channels = n_features, n_filters = n_filters_unet, activation = activation_unet)
      self.reconstruction = Reconstruction(in_channels=n_features)
      self.segmenter = Feature2Segmentation(in_channels=n_features, out_channels=self.n_classes)
    
    def forward(self,t2):

      ft2 = self.unet(t2)
      rt2 = self.reconstruction(ft2)
      st2 = self.segmenter(ft2)

      return ft2, rt2, st2

    def evaluate_batch(self, batch):
      patches_batch = batch
      t1 = patches_batch['t1'][tio.DATA]
      t2 = patches_batch['t2'][tio.DATA]
      s = patches_batch['label'][tio.DATA]

      ft2, rt2, st2 = self(t2)
      bce = nn.BCEWithLogitsLoss()
      #loss_recon = F.mse_loss(rt2,t2)
      #loss_seg = bce(st2,s)
      loss_recon = F.mse_loss(rt2,t1)
      loss_seg = 0

      return loss_recon, loss_seg

    def training_step(self, batch, batch_idx):
      loss_recon, loss_seg = self.evaluate_batch(batch)
      loss = loss_recon + loss_seg
      #loss = loss_seg

      self.log('train_loss', loss)
      self.log('train_recon_loss', loss_recon)
      self.log('train_seg_loss', loss_seg)
      return loss

    def validation_step(self, batch, batch_idx):
      loss_recon, loss_seg = self.evaluate_batch(batch)
      loss = loss_recon + loss_seg
      #loss = loss_seg
      self.log('val_loss', loss)
      self.log('val_recon_loss', loss_recon)
      self.log('val_seg_loss', loss_seg)
      self.log("hp_metric", loss)

      return loss

    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      return optimizer  


  if args.model is not None:
    net = Disentangler().load_from_checkpoint(args.model, n_features=n_features, n_filters_unet = n_filters_unet, activation_unet = activation_unet, learning_rate=learning_rate)
  else:
    net = Disentangler(n_features=n_features, n_filters_unet = n_filters_unet, activation_unet = activation_unet, learning_rate=learning_rate)

  checkpoint_callback = ModelCheckpoint(
    dirpath=output_path,
    filename=prefix+'_{epoch:02d}', 
    verbose=True
    )

  trainer = pl.Trainer(gpus=1, 
                       logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, name=prefix, log_graph=True),
                       max_epochs=num_epochs, 
                       progress_bar_refresh_rate=20, 
                       callbacks=[checkpoint_callback])
  if num_epochs > 0:
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

  output_keys = ['ft2','rt2','st2']

  aggregators = {}
  for k in output_keys:
    aggregators[k] = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

  net.eval()

  with torch.no_grad():
    for patches_batch in patch_loader:
      t1 = patches_batch['t1'][tio.DATA]
      t2 = patches_batch['t2'][tio.DATA]
      s = patches_batch['label'][tio.DATA]
      locations = patches_batch[tio.LOCATION]

      ft2, rt2, st2 = net(t2)

      aggregators['ft2'].add_batch(ft2, locations)  
      aggregators['rt2'].add_batch(rt2, locations)  
      aggregators['st2'].add_batch(st2, locations)

  print('Saving images...')
  for k in output_keys:
    output = aggregators[k].get_output_tensor()
    o = tio.ScalarImage(tensor=output, affine=subject['t1'].affine)
    o.save(output_path+prefix+'_'+k+'.nii.gz')

  subject['t2'].save(output_path+prefix+'_t2.nii.gz')
  subject['t1'].save(output_path+prefix+'_t1.nii.gz')
  subject['label'].save(output_path+prefix+'_gt.nii.gz')

  output_tensor = aggregators['st2'].get_output_tensor()
  output_tensor = torch.sigmoid(output_tensor)
  o = tio.ScalarImage(tensor=output_tensor, affine=subject['t1'].affine)
  o.save(output_path+prefix+'_seg.nii.gz')

  output_pred = torch.unsqueeze(torch.argmax(output_tensor,dim=0),0).int()
  output_seg = tio.LabelMap(tensor=output_pred, affine=subject['t1'].affine)
  output_seg.save(output_path+prefix+'_label.nii.gz')

