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
  parser.add_argument('-s', '--sigma', help='sigma value for BM3D denoising', type=float, required=False, default=0.025) 

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

  #Find automatically all images in denoised directory 
  denoised_stacks = []
  files = glob.glob(os.path.join(args.output,'denoised','*.nii.gz'))
  for f in files:
    if args.keyword in f:
      denoised_stacks.append(f)
  print('List of denoised stacks:')    
  print(denoised_stacks)      
  
  #fetal brain masking in raw images
  cmd_os = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_segment_fetal_brains '
  docker_output = os.path.join(args.output,'seg')
  docker_output = docker_output.replace('rousseau','data')
  cmd_os+= ' --dir-output '+docker_output  
  docker_stacks = [s.replace('rousseau','data') for s in denoised_stacks]
  docker_masks  = [s.replace('denoised','seg') for s in docker_stacks]

  cmd_os+= ' --filenames '
  for i in docker_stacks:
    cmd_os+= i+' '
  cmd_os+= ' --filenames-masks '
  for i in docker_masks:
    cmd_os+= i+' '

  print(cmd_os)
  #os.system(cmd_os)
  
  #reconstruction using niftymic
  cmd_os = 'docker run -it --rm --mount type=bind,source='+home+',target=/home/data renbem/niftymic niftymic_reconstruct_volume '
  docker_output = os.path.join(args.output,'recon')
  docker_output = docker_output.replace('rousseau','data')
  cmd_os+= ' --output '+os.path.join(docker_output,'niftymic_'+args.keyword+'_r05.nii.gz')+' --isotropic-resolution 0.5 '

  cmd_os+= ' --filenames '
  for i in docker_stacks:
    cmd_os+= i+' '
  cmd_os+= ' --filenames-masks '
  for i in docker_masks:
    cmd_os+= i+' '

  print(cmd_os)
  #os.system(cmd_os)

  #reconstruction using nesvor
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
  go = 'nesvor reconstruct --output-volume '+args.output+'/recon/nesvor_'+args.keyword+'_r6.nii.gz --output-resolution 6 --verbose 2 '
  go+= '--n-levels-bias 1 --single-precision '
  go+= '--inference-batch-size 2048 --n-inference-samples 512 '
  go+= '--output-model '+args.output+'/recon/nesvor_'+args.keyword+'_model.pt '
  go+= ' --input-stacks '
  for i in denoised_stacks:
    go+= i+' '
  mask_stacks =[s.replace('denoised','seg') for s in  denoised_stacks] 
  go+= ' --stack-masks '
  for i in mask_stacks:
    go+= i+' '
  print(go)
  os.system(go)    

  go = 'nesvor sample-volume --inference-batch-size 2048 --verbose 2 --output-volume '+args.output+'/recon/nesvor_'+args.keyword+'_r05.nii.gz --output-resolution 0.5 '
  go+= '--input-model '+args.output+'/recon/nesvor_'+args.keyword+'_model.pt '
  print(go)
  os.system(go)  