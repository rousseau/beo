#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse

from os.path import expanduser
home = expanduser("~")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo Fetal')
  parser.add_argument('-i', '--input', help='Input folder', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output folder', type=str, required=True)
  parser.add_argument('-k', '--keyword', help='Keyword used to select images', type=str, required=True)

  args = parser.parse_args()

  #mkdir seg et recon dans le folder output
  paths = ['seg','recon']
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

  go = 'niftymic_segment_fetal_brains --dir-output '+args.output
  go+= ' --filenames '
  for i in images:
    go+= i+' '
  go+= ' --filenames-masks '
  for i in images:
    go+= args.output+'/seg/'+i.split('/')[-1]+' '
  print(go)  
  #os.system(go)

  go = 'niftymic_reconstruct_volume --output '+args.output+'/recon/niftymic_'+args.keyword+'_r05.nii.gz --isotropic-resolution 0.5'
  go+= ' --filenames '
  for i in images:
    go+= i+' '
  go+= ' --filenames-masks '
  for i in images:
    go+= args.output+'/seg/'+i.split('/')[-1]+' '
  print(go)
  os.system(go)  