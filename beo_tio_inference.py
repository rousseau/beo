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

from beo_pl_nets import Unet

import glob

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

  args = parser.parse_args()


  subjects = []

  subject = tio.Subject(
        t2=tio.ScalarImage(args.input),
    )
  subjects.append(subject)

  dataset = tio.SubjectsDataset(subjects)
  print('Dataset size:', len(dataset), 'subjects')

  validation_transform = tio.Compose([
    #tio.ToCanonical(),
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

  net = Unet.load_from_checkpoint(args.model)
  net.eval()

  print('Inference')

  with torch.no_grad():
    for patches_batch in patch_loader:
      input_tensor = patches_batch['t2'][tio.DATA]
      locations = patches_batch[tio.LOCATION]
      outputs = net(input_tensor)
      aggregator.add_batch(outputs, locations)
  output_tensor = aggregator.get_output_tensor()
  output_tensor = torch.sigmoid(output_tensor)

  print(output_tensor.shape)
  print('Saving images') 

  output_pred = torch.unsqueeze(torch.argmax(output_tensor,dim=0),0).int()
  print(output_pred.shape)
  output_seg = tio.ScalarImage(tensor=output_pred, affine=subject['t2'].affine)
  output_seg.save(args.output)

  if args.fuzzy is not None:
    output_seg = tio.ScalarImage(tensor=output_tensor, affine=subject['t2'].affine)
    output_seg.save(args.fuzzy)  