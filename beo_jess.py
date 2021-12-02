#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
home = expanduser("~")

import torch
import torch.nn as nn 
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


import torchio as tio
import glob


in_channels = 3 #1 or 3

subjects = []

data_path = home+'/Sync-Exp/Data/DHCP/'
all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

max_subjects = 500
all_seg = all_seg[:max_subjects] 

for seg_file in all_seg:
  id_subject = seg_file.split('/')[6].split('_')[0:2]
  id_subject = id_subject[0]+'_'+id_subject[1]

  t2_file = [s for s in all_t2s if id_subject in s][0]
  t1_file = [s for s in all_t1s if id_subject in s][0]

  if in_channels == 1:  
    subject = tio.Subject(
      hr=tio.ScalarImage(t2_file),
      lr_1=tio.ScalarImage(t2_file),
    )
  if in_channels == 3:  
    subject = tio.Subject(
      hr=tio.ScalarImage(t2_file),
      lr_1=tio.ScalarImage(t2_file),
      lr_2=tio.ScalarImage(t2_file),
      lr_3=tio.ScalarImage(t2_file),
    )
  
  subjects.append(subject) 

print('DHCP Dataset size:', len(subjects), 'subjects')

# DATA AUGMENTATION
normalization = tio.ZNormalization()
spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)

tocanonical = tio.ToCanonical()

b1 = tio.Blur(std=(0.001,0.001,1), include='lr_1') #blur
d1 = tio.Resample((0.8,0.8,2), include='lr_1')     #downsampling
u1 = tio.Resample(target='hr', include='lr_1')     #upsampling

if in_channels == 3:
  b2 = tio.Blur(std=(0.001,1,0.001), include='lr_2') #blur
  d2 = tio.Resample((0.8,2,0.8), include='lr_2')     #downsampling
  u2 = tio.Resample(target='hr', include='lr_2')     #upsampling

  b3 = tio.Blur(std=(1,0.001,0.001), include='lr_3') #blur
  d3 = tio.Resample((2,0.8,0.8), include='lr_3')     #downsampling
  u3 = tio.Resample(target='hr', include='lr_3')     #upsampling

if in_channels == 1:
  transforms = [tocanonical, flip, spatial, normalization, b1, d1, u1]
  training_transform = tio.Compose(transforms)
  validation_transform = tio.Compose([tocanonical, normalization, b1, d1, u1])
if in_channels == 3:
  transforms = [tocanonical, flip, spatial, normalization, b1, d1, u1, b2, d2, u2, b3, d3, u3]
  training_transform = tio.Compose(transforms)
  validation_transform = tio.Compose([tocanonical, normalization, b1, d1, u1, b2, d2, u2, b3, d3, u3])


# SPLIT DATA
seed = 42  # for reproducibility
training_split_ratio = 0.9
num_subjects = len(subjects)
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
num_workers = 16
print('num_workers : '+str(num_workers))
patch_size = 96
max_queue_length = 1024
samples_per_volume = 8
batch_size = 1

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

training_loader_patches = torch.utils.data.DataLoader(
  patches_training_set, batch_size=batch_size)

validation_loader_patches = torch.utils.data.DataLoader(
  patches_validation_set, batch_size=batch_size)

#%%
#Choisir un reseau Monai
class ResNetBlock(torch.nn.Module):
  def __init__(self, in_channels = 32):
    super(ResNetBlock, self).__init__()
    self.in_channels = in_channels
    def double_conv(in_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_channels),
      )
    self.dc = double_conv(self.in_channels)
  
  def forward(self,x):
    z = self.dc(x)
    return x+z

class ResNet(torch.nn.Module):
  def __init__(self, in_channels = 1, out_channels = 10, n_filters = 32, n_layers = 5):
    super(ResNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_features = n_filters
    self.n_layers = n_layers

    self.blocks = torch.nn.ModuleList()
    for i in range(n_layers):
      self.blocks.append(ResNetBlock(in_channels = self.n_features))
      
    self.inconv = nn.Conv3d(self.in_channels, self.n_features, kernel_size=3, padding=1)
    self.outconv = nn.Conv3d(self.n_features, self.out_channels, kernel_size=3, padding=1)

  def forward(self,x):
    z = self.inconv(x)
    z = nn.ReLU()(z)
    for i in range(self.n_layers):
      z = self.blocks[i](z)
    z = self.outconv(z)
    return z  


class Net(pl.LightningModule):
  def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, activation = 'relu'):
    super(Net, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_features = n_filters

    self.net = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = 10)

  def forward(self, x):
    return self.net(x)

  def evaluate_batch(self, batch):
    patches_batch = batch

    if self.in_channels == 1:
      hr = patches_batch['hr'][tio.DATA]
      lr = patches_batch['lr_1'][tio.DATA]
    if self.in_channels == 3:  
      hr = patches_batch['hr'][tio.DATA]
      lr_1 = patches_batch['lr_1'][tio.DATA]
      lr_2 = patches_batch['lr_2'][tio.DATA]
      lr_3 = patches_batch['lr_3'][tio.DATA]
      lr = torch.concat((lr_1,lr_2,lr_3),1)

    rlr = self(lr)
    loss_recon = F.mse_loss(rlr,hr)

    return loss_recon
      
  def training_step(self, batch, batch_idx):
    loss = self.evaluate_batch(batch)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    loss = self.evaluate_batch(batch)
    self.log('val_loss', loss)
    self.log("hp_metric", loss)
    return loss

  def predict_step(self, batch, batch_idx, dataloader_idx=0):
    return self(batch)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer  



import argparse
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO inference')
  parser.add_argument('-e', '--epochs', help='Number of epochs (for training)', type=int, required=False, default = 10)
  parser.add_argument('-i', '--input', action='append', help='Input image for inference (axial, coronal, sagittal)', type=str, required=False)
  parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=False)
  #parser.add_argument('-o', '--output', help='Output image', type=str, required=False)

  args = parser.parse_args()
  
  num_epochs = args.epochs
  output_path = home+'/Sync-Exp/Experiments/'
  prefix = 'resnet_nl10_recon'
  prefix+= '_ic'+str(in_channels)
  prefix+= '_e'+str(num_epochs)

  patch_overlap = int(patch_size / 2)  

  if num_epochs > 0:
    if args.model is not None:
      print('Loading model.')
      net = Net.load_from_checkpoint(args.model)
    else:  
      net = Net(in_channels = in_channels, out_channels = 1, n_filters = 32, activation = 'relu')
    
    checkpoint_callback = ModelCheckpoint(
      dirpath=output_path,
      filename=prefix+'_{epoch:02d}', 
      verbose=True
      )
    trainer = pl.Trainer(gpus=1, 
                          logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, name=prefix, log_graph=True),
                          max_epochs=num_epochs, 
                          progress_bar_refresh_rate=20, 
                          callbacks=[checkpoint_callback],
                          benchmark=True,
                          amp_level='01',
                          auto_scale_batch_size="binsearch")

    trainer.fit(net, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(output_path+prefix+'.ckpt')  
    print('Finished Training')  

  else:
      net = Net.load_from_checkpoint(args.model)
      

#%%
  print('Inference')
  net.eval()
  if num_epochs > 0:
    subject = validation_set[0]
  else:
    if in_channels == 1:
      input_subject = tio.Subject(
        lr_1=tio.ScalarImage(args.input[0]),
      )
    if in_channels == 3:
      input_subject = tio.Subject(
        lr_1=tio.ScalarImage(args.input[0]),
        lr_2=tio.ScalarImage(args.input[1]),
        lr_3=tio.ScalarImage(args.input[2]),                
      )

    transforms = tio.Compose([tocanonical, normalization]) 
    subject = transforms(input_subject)

  grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
    )

  patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

  output_keys = ['rlr']

  aggregators = {}
  for k in output_keys:
    aggregators[k] = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

  with torch.no_grad():
    for patches_batch in patch_loader:
      if in_channels == 1:
        lr = patches_batch['lr_1'][tio.DATA]
      if in_channels == 3:  
        lr_1 = patches_batch['lr_1'][tio.DATA]
        lr_2 = patches_batch['lr_2'][tio.DATA]
        lr_3 = patches_batch['lr_3'][tio.DATA]
        lr = torch.concat((lr_1,lr_2,lr_3),1)

      locations = patches_batch[tio.LOCATION]

      rlr = net(lr)

      aggregators['rlr'].add_batch(rlr, locations)  

  print('Saving images...')
  for k in output_keys:
    output = aggregators[k].get_output_tensor()
    o = tio.ScalarImage(tensor=output, affine=subject['lr'].affine)
    o.save(output_path+prefix+'_'+k+'.nii.gz')

  #subject['hr'].save(output_path+prefix+'_hr.nii.gz')
  subject['lr'].save(output_path+prefix+'_lr.nii.gz')      