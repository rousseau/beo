#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from beo_wrappers import wrapper_scunet, wrapper_masking, wrapper_reconstruction, wrapper_svrtk_reorientation, wrapper_svrtk_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo denoising for a given directory')
    parser.add_argument('-i', '--input', help='Input folder (absolute path)', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output folder (absolute path)', type=str, required=True)
 
    args = parser.parse_args()
 
    #Find automatically all images in input directory
    raw_stacks = []
    files = glob.glob(os.path.join(args.input,'*.nii.gz'))
    for f in files:
        raw_stacks.append(f)
    print('List of input raw images:')    
    print(raw_stacks) 

    # check is output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # DENOISING
    denoised_stacks = []
    for file in raw_stacks:
        outputfile = os.path.join(args.output, os.path.basename(file).replace('.nii.gz','_denoised.nii.gz'))
        denoised_stacks.append(outputfile)
        if not os.path.exists(outputfile):
            wrapper_scunet(file, outputfile)
