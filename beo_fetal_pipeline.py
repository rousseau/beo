#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os.path import expanduser
home = expanduser("~")
import glob
import argparse
import nibabel
import numpy as np
import bm3d

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo Fetal using Nesvor')
  parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output folder', type=str, required=True)
  parser.add_argument('-k', '--keyword', help='Keyword used to select images (like HASTE ou TrueFISP)', type=str, required=True)

  parser.add_argument('-s', '--sigma', help='sigma value for BM3D denoising', type=float, required=0.05)
  # 0.025 TrueFISP B
  # 
  args = parser.parse_args()


  #mkdir dans le folder output
  paths = ['denoised','seg','recon']
  for p in paths:
    new_path = os.path.join(args.output,p)
    if not os.path.exists(new_path):
      os.makedirs(new_path)

  #Find automatically all images in input directory 
  raw_stacks = []
  files = glob.glob(os.path.join(args.input,'*.nii.gz'))
  for f in files:
    if args.keyword in f:
      raw_stacks.append(f)
  print('List of input raw stacks:')    
  print(raw_stacks)      

  #denoising using bm3d
  for f in raw_stacks:
    print(f)
    image= nibabel.load(f)
    data = image.get_fdata()
    dmax = np.max(data)
    sigma = args.sigma
    #bm3d_denoised = dmax * bm3d.bm3d(data/dmax, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    outputfile = os.path.join(args.output,'denoised',f.split('/')[-1])
    #nibabel.save(nibabel.Nifti1Image(bm3d_denoised, image.affine), outputfile) 

  #fetal brain masking in raw images
  cmd_os = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_segment_fetal_brains '
  cmd_os+= ''  
  print(cmd_os)
  