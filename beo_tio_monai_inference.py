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

import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo TorchIO inference')
  parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
  parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
  parser.add_argument('-f', '--fuzzy', help='Output fuzzy image', type=str, required=False)
  parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = 128)
  parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = 64)
  parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
  parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
  parser.add_argument('-t', '--test_time', help='Number of inferences for test-time augmentation', type=int, required=False, default=1)
  parser.add_argument('-c', '--channels', help='Number of channels', type=int, required=False, default=16)
  parser.add_argument('-l', '--levels', help='Number of levels in Unet (up to 5)', type=int, required=False, default=3)
  parser.add_argument('--classes', help='Number of classes', type=int, required=False, default=10)
  parser.add_argument('-s', '--scales', help='Scaling factor (test-time augmentation)', type=float, required=False, default=0.05)
  parser.add_argument('-d', '--degrees', help='Rotation degrees (test-time augmentation)', type=int, required=False, default=10)

  args = parser.parse_args()
  n_levels = args.levels
  n_channels = args.channels

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
  #%%
  unet = monai.networks.nets.UNet(
      dimensions=3,
      in_channels=1,
      out_channels=args.classes,
      channels=channels_tuple,
      strides=(2, 2, 2),
      num_res_units=2,
  )
  unet.load_state_dict(torch.load(args.model))


  class Model(pl.LightningModule):
      def __init__(self, net):
          super().__init__()
          self.net = net
      
      def forward(self,x):
          return self.net(x)
  
  net = Model(net=unet)
  net.eval()
  #%%

  subject = tio.Subject(
        image=tio.ScalarImage(args.input),
    )

  #%%
  normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
  onehot = tio.OneHot()
  spatial = tio.RandomAffine(scales=args.scales,degrees=args.degrees,translation=5,image_interpolation='bspline',p=1)
  flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)

  print('Inference')

  patch_overlap = args.patch_overlap
  patch_size = args.patch_size
  batch_size = args.batch_size

  output_tensors = []

  for i in range(args.test_time):
    if i == 0:
      augment = normalization
    else:
      augment = tio.Compose((normalization,flip, spatial))

    subj = augment(subject)

    print(subj.get_composed_history())

    grid_sampler = tio.inference.GridSampler(
      subj,
      patch_size,
      patch_overlap,
      )

    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

    with torch.no_grad():
      for patches_batch in patch_loader:
        input_tensor = patches_batch['image'][tio.DATA]
        locations = patches_batch[tio.LOCATION]
        outputs = net(input_tensor)
        aggregator.add_batch(outputs, locations)
    output_tensor = aggregator.get_output_tensor()
    output_tensor = torch.sigmoid(output_tensor)

    tmp = tio.ScalarImage(tensor=output_tensor, affine=subj.image.affine)
    subj.add_image(tmp, 'label')

    back = subj.apply_inverse_transform(image_interpolation='linear')

    output_tensors.append(torch.unsqueeze(back.label.data,0))

  output_tensor = torch.squeeze(torch.stack(output_tensors, dim=0).mean(dim=0))
  print(output_tensor.shape)

  print('Saving images') 

  output_pred = torch.unsqueeze(torch.argmax(output_tensor,dim=0),0).int()
  print(output_pred.shape)
  output_seg = tio.LabelMap(tensor=output_pred, affine=subject['image'].affine)
  output_seg.save(args.output)

  if args.fuzzy is not None:
    output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['image'].affine)
    output_seg.save(args.fuzzy)  

  if args.ground_truth is not None:
    gt_image= onehot(tio.LabelMap(args.ground_truth))
    pred_image = onehot(output_seg)
    
    print(gt_image.data.shape)
    print(pred_image.data.shape)

    dice_val = monai.metrics.compute_meandice(torch.unsqueeze(pred_image.data,0), torch.unsqueeze(gt_image.data,0), include_background=True)

    print("DICE :")
    print(dice_val)
