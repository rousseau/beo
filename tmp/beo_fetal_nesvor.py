#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse

from os.path import expanduser
home = expanduser("~")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo Fetal using Nesvor')
  parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output folder', type=str, required=True)
  parser.add_argument('-k', '--keyword', help='Keyword used to select images', type=str, required=True)

  args = parser.parse_args()

  #Get images with masks (previously computed using Nitfymic)
  images = []
  files = glob.glob(args.output+'/seg/*.nii.gz')
  for f in files:
    if args.keyword in f:
      images.append(args.input+'/'+f.split('/')[-1])
  print(images)

  go = 'nesvor reconstruct --output-volume '+args.output+'/recon/nesvor_'+args.keyword+'_r6.nii.gz --output-resolution 6 '
  go+= '--inference-batch-size 4000 --n-inference-samples 512 '
  go+= '--output-model '+args.output+'/recon/nesvor_'+args.keyword+'_model.pt '
  go+= ' --input-stacks '
  for i in images:
    go+= i+' '
  go+= ' --stack-masks '
  for i in images:
    go+= args.output+'/seg/'+i.split('/')[-1]+' '
  print(go)
  os.system(go)  

  go = 'nesvor sample-volume --output-volume '+args.output+'/recon/nesvor_'+args.keyword+'_r05.nii.gz --output-resolution 0.5 '
  go+= '--input-model '+args.output+'/recon/nesvor_'+args.keyword+'_model.pt '
  print(go)
  os.system(go)
