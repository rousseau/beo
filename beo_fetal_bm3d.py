#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import expanduser
home = expanduser("~")
import nibabel
import bm3d
import numpy as np
import glob
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo Fetal using Nesvor')
  parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output folder', type=str, required=True)
  parser.add_argument('-k', '--keyword', help='Keyword used to select images', type=str, required=True)

  args = parser.parse_args()

  #mkdir denoised dans le folder output
  paths = ['denoised']
  for p in paths:
    if not os.path.exists(args.output+'/'+p):
      os.makedirs(args.output+'/'+p)

  #Find automatically all images in current directory (remove images with 'mask' or 'crop' in filename)
  images = []
  files = glob.glob(args.input+'/*.nii.gz')
  for f in files:
    if args.keyword in f:
      images.append(f)
  print(images)

  for f in images:
    print(f)
    image= nibabel.load(f)
    data = image.get_fdata()
    dmax = np.max(data)
    sigma = 0.05
    bm3d_denoised = dmax * bm3d.bm3d(data/dmax, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    outputfile = args.output+'/denoised/'+f.split('/')[-1]
    nibabel.save(nibabel.Nifti1Image(bm3d_denoised, image.affine), outputfile) 