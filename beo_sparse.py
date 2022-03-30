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
import numpy as np

in_channels = 1

subjects = []

data_path = home+'/Sync-Exp/Data/DHCP/'
all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
all_t2s = glob.glob(data_path+'**/*desc-restore_T2w.nii.gz', recursive=True)
all_t1s = glob.glob(data_path+'**/*desc-restore_space-T2w_T1w.nii.gz', recursive=True)

max_subjects = 400
all_seg = all_seg[:max_subjects] 

alpha = 0.05 #probability to keep data points

for seg_file in all_seg:
  id_subject = seg_file.split('/')[6].split('_')[0:2]
  id_subject = id_subject[0]+'_'+id_subject[1]

  t2_file = [s for s in all_t2s if id_subject in s][0]

  subject = tio.Subject(
    hr=tio.ScalarImage(t2_file),
  )

  shape = subject.hr.shape
  mask = torch.FloatTensor(np.random.choice([0, 1], size=shape, p=[1-alpha, alpha]))
  lr = mask * subject.hr.data
  subject.add_image(tio.LabelMap(tensor=mask, affine=subject.hr.affine), "mask")
  subject.add_image(tio.ScalarImage(tensor=lr, affine=subject.hr.affine), "lr")
  
  subjects.append(subject) 

print('DHCP Dataset size:', len(subjects), 'subjects')

# DATA AUGMENTATION
normalization = tio.ZNormalization()
flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)

tocanonical = tio.ToCanonical()
transforms = [tocanonical, flip, normalization]
training_transform = tio.Compose(transforms)
validation_transform = tio.Compose([tocanonical, normalization])

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

num_workers = 2
print('num_workers : '+str(num_workers))
patch_size = 96
max_queue_length = 1024
samples_per_volume = 8
batch_size = 4

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
    def block(in_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_channels),
      )
    self.dc = block(self.in_channels)
  
  def forward(self,x):
    z = self.dc(x)
    return x+z

class DoubleConvBlock(torch.nn.Module):
  def __init__(self, in_channels = 32):
    super(DoubleConvBlock, self).__init__()
    self.in_channels = in_channels
    def double_conv(in_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
      )
    self.dc = double_conv(self.in_channels)
  
  def forward(self,x):
    z = self.dc(x)
    return z

class ResNet(torch.nn.Module):
  def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, n_layers = 5):
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

class UNet(torch.nn.Module):
  def __init__(self, in_channels = 1, out_channels = 1, n_features = 32):
    super(UNet, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_features = n_features

    self.dc1 = ResNet(in_channels = 1, out_channels = 1, n_filters = 16, n_layers = 5)
    self.dc2 = ResNet(in_channels = 2, out_channels = 1, n_filters = 16, n_layers = 5)
    self.dc3 = ResNet(in_channels = 2, out_channels = 1, n_filters = 16, n_layers = 5)

    self.mp = nn.MaxPool3d(2)
    self.up = nn.Upsample(scale_factor=2, mode='nearest')#, align_corners=True)

    self.out = nn.Conv3d(self.n_features, self.out_channels, kernel_size=1)
    self.conv_up = nn.Conv3d(1, 1, kernel_size=3, padding=1)

    self.tmp = nn.Conv3d(1, 1, kernel_size=3, padding=1)

  def forward(self, x):
    #level 0 : original size
    #level 1 : downsampled by 2
    #level 2 : downsampled by 4

    x_l0 = x
    x_l1 = self.mp(x_l0)
    x_l2 = self.mp(x_l1)

    x_l2_out = self.dc1(x_l2) #resnet to compute first estimate at scale l2
    x_l2_up = self.conv_up(self.up(x_l2_out)) #upsampling to l1

    x_l1_concat = torch.cat([x_l2_up,x_l1],dim=1) #concat estimate and data at scale l1
    x_l1_out = self.dc2(x_l1_concat)        #resnet at scale l1
    x_l1_up = self.conv_up(self.up(x_l1_out)) #upsampling to l0

    x_l0_concat = torch.cat([x_l1_up,x_l0],dim=1) #concat estimate and data at scale l0
    x_l0_out = self.dc3(x_l0_concat)        #resnet at scale l0

    return x_l0_out, x_l1_out, x_l2_out   


class Net(pl.LightningModule):
  def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32, activation = 'relu'):
    super(Net, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_features = n_filters

    #self.net = ResNet(in_channels = in_channels, out_channels = out_channels, n_filters = n_filters, n_layers = 10)
    self.net = UNet(in_channels = in_channels, out_channels = out_channels, n_features = n_filters)

    self.save_hyperparameters()

  def forward(self, x):
    return self.net(x)

  def evaluate_batch(self, batch):
    patches_batch = batch

    if self.in_channels == 1:
      hr = patches_batch['hr'][tio.DATA]
      lr = patches_batch['lr'][tio.DATA]

    rlr, rlr_1, rlr_2 = self(lr)
    hr_1 = nn.functional.avg_pool3d(hr,2)
    hr_2 = nn.functional.avg_pool3d(hr_1,2)    
    loss_recon = F.mse_loss(rlr,hr) + F.mse_loss(rlr_1,hr_1) + F.mse_loss(rlr_2,hr_2)

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
  prefix = 'sparse_unet_recon'
  prefix+= '_ic'+str(in_channels)
  prefix+= '_e'+str(num_epochs)
  prefix+= '_a'+str(alpha)
  prefix+= '_s'+str(len(subjects))

  patch_overlap = int(patch_size / 2)  

  if num_epochs > 0:
    if args.model is not None:
      print('Loading model.')
      net = Net.load_from_checkpoint(args.model)
    else:  
      net = Net(in_channels = in_channels, out_channels = 1, n_filters = 16, activation = 'relu')
    
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
                          #amp_level='01',
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
    input_subject = tio.Subject(
      lr=tio.ScalarImage(args.input[0]),
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

  device = torch.device('cuda:0')
  net.to(device)

  with torch.no_grad():
    for patches_batch in patch_loader:
      lr = patches_batch['lr'][tio.DATA]

      locations = patches_batch[tio.LOCATION]

      lr = lr.to(device)
      rlr,_,_ = net(lr)

      aggregators['rlr'].add_batch(rlr.cpu(), locations)  

  print('Saving images...')
  for k in output_keys:
    output = aggregators[k].get_output_tensor()
    o = tio.ScalarImage(tensor=output, affine=subject['lr'].affine)
    o.save(output_path+prefix+'_'+k+'.nii.gz')

  if num_epochs > 0:
    subject['hr'].save(output_path+prefix+'_hr.nii.gz')
  subject['lr'].save(output_path+prefix+'_lr.nii.gz')      
