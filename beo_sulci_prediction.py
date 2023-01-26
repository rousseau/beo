#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import expanduser
home = expanduser("~")

import numpy as np
from scipy.ndimage import distance_transform_cdt, gaussian_filter, binary_fill_holes
import nibabel

import torch
import torch.nn as nn 
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torchio as tio
import glob

import pandas as pd




def sdf(x):
  sdf_in = distance_transform_cdt(x)
  sdf_out = distance_transform_cdt(1-x)
  sdf = (sdf_out-sdf_in)*1.0 #/ np.max(np.abs(sdf_in))
  sdf = gaussian_filter(sdf,1)
  sdf = np.clip(sdf,-20,20)
  return sdf
     



#%%

class Unet(nn.Module):
  def __init__(self, n_channels = 2, n_classes = 3, n_features = 8):
    super(Unet, self).__init__()

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.n_features = n_features

    def double_conv(in_channels, out_channels):
      return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
      )

    self.dc1 = double_conv(self.n_channels, self.n_features)
    self.dc2 = double_conv(self.n_features, self.n_features)
    self.dc3 = double_conv(self.n_features, self.n_features)
    self.dc4 = double_conv(self.n_features*2, self.n_features)
    self.dc5 = double_conv(self.n_features*2, self.n_features)
    
    self.ap = nn.AvgPool3d(2)

    self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    self.x3_out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)
    self.x4_out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

    self.out = nn.Conv3d(self.n_features, self.n_classes, kernel_size=1)

  def forward(self, x):
    x1 = self.dc1(x)

    x2 = self.ap(x1)
    x2 = self.dc2(x2)

    x3 = self.ap(x2)
    x3 = self.dc3(x3)

    x4 = self.up(x3)
    x4 = torch.cat([x4,x2], dim=1)
    x4 = self.dc4(x4)

    x5 = self.up(x4)
    x5 = torch.cat([x5,x1], dim=1)
    x5 = self.dc5(x5)
    return self.out(x5)

#%%
class Net(pl.LightningModule):
  def __init__(self, in_channels = 1, out_channels = 1, n_filters = 32):
    super(Net, self).__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_features = n_filters

    self.net = Unet(n_channels = in_channels, n_classes = out_channels, n_features = n_filters)
    self.save_hyperparameters()

  def forward(self, x):
    return self.net(x)

  def evaluate_batch(self, batch):
    patches_batch = batch

    hull = patches_batch['hull'][tio.DATA]
    brain = patches_batch['brain'][tio.DATA]
 
    #Select area to compute the loss
    pred = torch.where(torch.abs(brain) < 0.25 , self(hull), 0.)
    brain = torch.where(torch.abs(brain) < 0.25 , brain, 0.)

    return F.mse_loss(brain,pred)

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

#%%
import argparse
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo Sulci Learning')
  parser.add_argument('-e', '--epochs', help='Number of epochs (for training)', type=int, required=False, default = 10)
  parser.add_argument('-i', '--input', help='Input image for inference ', type=str, required=False)
  parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=False)
  parser.add_argument('-a', '--age_max', help='Maximum age', type=int, required=False, default = 40)
  #parser.add_argument('-o', '--output', help='Output image', type=str, required=False)

  args = parser.parse_args()

  subjects = []
  data_path = home+'/Sync-Exp/Data/DHCP/'
  csv_file = pd.read_csv(data_path+'combined.tsv',sep="\t",on_bad_lines='skip')
    
  max_subjects = 500
  num_workers = 8
  print('num_workers : '+str(num_workers))

  max_queue_length = 1024
  samples_per_volume = 16
  resolution = 1

  if resolution == 1:
    patch_size = 96
    batch_size = 4
  if resolution == 2:
    patch_size = 48
    batch_size = 32

  compute_sdf = False

  if compute_sdf is True:
    all_seg = glob.glob(data_path+'**/*fusion_space-T2w_dseg.nii.gz', recursive=True)
      
    for seg_file in all_seg:
      id_subject = seg_file.split('/')[6].split('_')[0:2]
      id_subject = id_subject[0]+'_'+id_subject[1]

      seg = nibabel.load(seg_file)
      seg_data = seg.get_fdata()

      hull = np.where(seg_data>0,1,0).astype(np.intc)
      hull = binary_fill_holes(hull).astype(np.intc)
      hull = binary_fill_holes(hull).astype(np.intc)
      
      #remove everything outside (also label 2 which is cortical gray matter)
      brain = (np.where(seg_data>0,1,0) - np.where(seg_data==1,1,0) - np.where(seg_data==4,1,0) - np.where(seg_data==6,1,0) - np.where(seg_data==2,1,0)).astype(np.intc)
      brain = binary_fill_holes(brain).astype(np.intc)
      brain = binary_fill_holes(brain).astype(np.intc)

      fileout_hull = data_path+'/'+id_subject+'_hull_sdf.nii.gz'
      fileout_brain = data_path+'/'+id_subject+'_brain_sdf.nii.gz'
      
      nibabel.save( nibabel.Nifti1Image(sdf(hull), seg.affine), fileout_hull)
      nibabel.save( nibabel.Nifti1Image(sdf(brain), seg.affine), fileout_brain)

      #Resample to speed up experiments
      res = str(resolution)+'x'+str(resolution)+'x'+str(resolution)
      go = 'ResampleImage 3 '+fileout_hull+' '+fileout_hull+' '+res+' 0 0 '
      os.system(go)
      go = 'ResampleImage 3 '+fileout_brain+' '+fileout_brain+' '+res+' 0 0 '
      os.system(go)
    

  all_hull_sdf = glob.glob(data_path+'**/*hull_sdf.nii.gz', recursive=True)
  all_brain_sdf = glob.glob(data_path+'**/*brain_sdf.nii.gz', recursive=True)

  all_hull_sdf = all_hull_sdf[:max_subjects]

  #Select interesting subjects using csv
  csv_subjects = []
  n_csv = len(csv_file[csv_file.keys()[0]])
  age_min = 0
  age_max = args.age_max
  for i in range(n_csv):
    scan_age = csv_file['scan_age'][i]
    if scan_age > age_min and scan_age < age_max:
      id_subject = 'sub-'+str(csv_file['participant_id'][i])+'_ses-'+str(csv_file['session_id'][i])
      csv_subjects.append(id_subject)

  print(csv_subjects)
  print(len(csv_subjects))

  for s in csv_subjects:

    hull_file = data_path+s+'_hull_sdf.nii.gz'
    brain_file= data_path+s+'_brain_sdf.nii.gz'

    if hull_file in all_hull_sdf:
      subject = tio.Subject(
          hull=tio.ScalarImage(hull_file),
          brain=tio.ScalarImage(brain_file),
        )
    
      subjects.append(subject) 


  """ 
  for hull_file in all_hull_sdf:
    id_subject = hull_file.split('/')[6].split('_')[0:2]
    id_subject = id_subject[0]+'_'+id_subject[1]

    brain_file = [s for s in all_brain_sdf if id_subject in s][0]

    subject = tio.Subject(
        hull=tio.ScalarImage(hull_file),
        brain=tio.ScalarImage(brain_file),
      )
    
    subjects.append(subject) 
  """
  print('DHCP Dataset size:', len(subjects), 'subjects')

  #%% DATA AUGMENTATION
  #normalization = tio.ZNormalization()
  #SDF image intensities lie in [-20,20]
  normalization = tio.RescaleIntensity(out_min_max=(-1, 1))
  spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
  tocanonical = tio.ToCanonical()

  transforms = [tocanonical, flip, spatial, normalization]
  training_transform = tio.Compose(transforms)
  validation_transform = tio.Compose([tocanonical, normalization])

  #%%

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
  
  num_epochs = args.epochs
  output_path = home+'/Sync-Exp/Experiments/'
  prefix = 'sulci_prediction'
  prefix+= '_e'+str(num_epochs)
  prefix+= '_r'+str(resolution)
  prefix+= '_a'+str(age_max)

  patch_overlap = int(patch_size / 2)  

  if args.model is not None:
    print('Loading model.')
    net = Net.load_from_checkpoint(args.model)
  else:  
    net = Net(in_channels = 1, out_channels = 1, n_filters = 32)

  if num_epochs > 0:
    checkpoint_callback = ModelCheckpoint(
      dirpath=output_path,
      filename=prefix+'_{epoch:02d}', 
      verbose=True
      )
    trainer = pl.Trainer(gpus=1, 
                          logger=TensorBoardLogger(save_dir='lightning_logs', default_hp_metric=False, name=prefix, log_graph=True),
                          max_epochs=num_epochs, 
                          callbacks=[checkpoint_callback],
                        )

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
      hull=tio.ScalarImage(args.input[0]),
    )

    #transforms = tio.Compose([tocanonical, normalization]) 
    subject = validation_transform(input_subject)

  grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
    )

  patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)

  aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

  device = torch.device('cuda:0')
  net.to(device)

  with torch.no_grad():
    for patches_batch in patch_loader:
      hull = patches_batch['hull'][tio.DATA]
      locations = patches_batch[tio.LOCATION]

      hull = hull.to(device)
      brain = net(hull)

      aggregator.add_batch(brain.cpu(), locations)  

  print('Saving images...')
  output = aggregator.get_output_tensor()
  o = tio.ScalarImage(tensor=output, affine=subject['hull'].affine)
  o.save(output_path+prefix+'_pred.nii.gz')

  if num_epochs > 0:
    subject['brain'].save(output_path+prefix+'_brain.nii.gz')
  subject['hull'].save(output_path+prefix+'_hull.nii.gz')      
