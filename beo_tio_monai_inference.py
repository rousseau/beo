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

  args = parser.parse_args()


  subjects = []

  subject = tio.Subject(
        image=tio.ScalarImage(args.input),
    )
  subjects.append(subject)

  dataset = tio.SubjectsDataset(subjects)
  print('Dataset size:', len(dataset), 'subjects')

  validation_transform = tio.Compose([
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    tio.OneHot(),
  ])


  validation_set = tio.SubjectsDataset(
    subjects, transform=validation_transform )

  print('Patch sampling')

  patch_overlap = args.patch_overlap
  patch_size = args.patch_size
  batch_size = args.batch_size

  subject = validation_set[0]

  grid_sampler = tio.inference.GridSampler(
    subject,
    patch_size,
    patch_overlap,
    )

  patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
  aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

  unet = monai.networks.nets.UNet(
      dimensions=3,
      in_channels=1,
      out_channels=10,
      channels=(16, 32, 64),
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
  
  net = Model(
    net=unet
  )
  net.eval()

  print('Inference')

  with torch.no_grad():
    for patches_batch in patch_loader:
      input_tensor = patches_batch['image'][tio.DATA]
      locations = patches_batch[tio.LOCATION]
      outputs = net(input_tensor)
      aggregator.add_batch(outputs, locations)
  output_tensor = aggregator.get_output_tensor()
  output_tensor = torch.sigmoid(output_tensor)

  print(output_tensor.shape)
  print('Saving images') 

  output_pred = torch.unsqueeze(torch.argmax(output_tensor,dim=0),0).int()
  print(output_pred.shape)
  output_seg = tio.ScalarImage(tensor=output_pred, affine=subject['image'].affine)
  output_seg.save(args.output)

  if args.fuzzy is not None:
    output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['image'].affine)
    output_seg.save(args.fuzzy)  

  if args.ground_truth is not None:
    gt_image= validation_transform(tio.LabelMap(args.ground_truth))
    pred_image = validation_transform(output_seg)

    dice_val = monai.metrics.compute_meandice(torch.unsqueeze(pred_image.data,0), torch.unsqueeze(gt_image.data,0), include_background=True)

    print("DICE :")
    print(dice_val)

