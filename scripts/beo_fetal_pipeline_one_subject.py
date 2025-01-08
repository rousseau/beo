#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from beo_wrappers import wrapper_scunet, wrapper_masking, wrapper_reconstruction, wrapper_svrtk_reorientation, wrapper_svrtk_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo fetal pipeline of one subject')
    parser.add_argument('-i', '--input', help='Input folder (absolute path)', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output folder (absolute path)', type=str, required=True)
    parser.add_argument('-k', '--keyword', help='Keyword used to select images (like HASTE ou TrueFISP)', type=str, required=True)
    parser.add_argument('-m', '--masking', help='Masking method (nesvor, niftymic, synthstrip)', type=str, required=False, default='nesvor')
    parser.add_argument('-r', '--recon', help='Reconstruction method (nesvor, niftymic, svrtk)', type=str, required=False, default='nesvor')

    args = parser.parse_args()
 
    #Find automatically all images in input directory
    raw_stacks = []
    files = glob.glob(os.path.join(args.input,'*.nii.gz'))
    for f in files:
        if args.keyword in f:
            raw_stacks.append(f)
    print('List of input raw stacks:')    
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

    # MASKING
    mask_stacks = []
    for file in denoised_stacks:
        outputfile = os.path.join(args.output, os.path.basename(file).replace('.nii.gz','_mask.nii.gz'))
        mask_stacks.append(outputfile)
        if not os.path.exists(outputfile):
            wrapper_masking(file, outputfile,method=args.masking)    

    # RECONSTRUCTION
    recon_file = args.output + '/reconstruction_'+args.recon+'.nii.gz'
    if not os.path.exists(recon_file):
        wrapper_reconstruction(denoised_stacks, mask_stacks, recon_file, method=args.recon)

    # REORIENTATION
    reo_file = args.output + '/reconstruction_'+args.recon+'_reo.nii.gz'
    if not os.path.exists(reo_file):
        wrapper_svrtk_reorientation(recon_file, reo_file)

    # SEGMENTATION
    seg_file = args.output + '/segmentation_'+args.recon+'.nii.gz'
    wrapper_svrtk_segmentation(reo_file, seg_file)
