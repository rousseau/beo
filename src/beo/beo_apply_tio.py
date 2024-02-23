#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import torchio as tio

#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply torchio transforms on nifti images.')
    parser.add_argument('-i', '--input_image', help='Input nifti image ', type=str, required = True)
    parser.add_argument('-o', '--output_image', help='Output nifti image ', type=str, required = True)

    parser.add_argument('-m', '--mask', help='Mask nifti image ', type=str, required = False)
    parser.add_argument('-l', '--label', help='Label nifti image ', type=str, required = False)

    parser.add_argument('--rescale', help='Rescale intensity', action='store_true')
    parser.add_argument('--normalization', help='Z normalization', action='store_true')
    parser.add_argument('--masking', help='Apply Mask', action='store_true')
    parser.add_argument('--croporpad', help='Image size for Crop or pad', type=int, required=False, default=0)
    parser.add_argument('--canonical', help='To canonical', action='store_true')
    parser.add_argument('--resample', help='Voxel size for Resample', type=float, required=False, default=0)
    parser.add_argument('--resize', help='Image size for Resize', type=float, required=False, default=0)

    args = parser.parse_args()
    print(args)

    sub_dict = {}
    arg_dict = {}
    
    sub_dict['image'] = tio.ScalarImage(args.input_image)
    if args.mask:
        sub_dict['mask'] = tio.LabelMap(args.mask)
    if args.label:
        sub_dict['label'] = tio.LabelMap(args.label)    


    subject = tio.Subject(sub_dict)

    transforms = []

    if args.rescale:
        if args.masking:
            transforms.append(tio.transforms.RescaleIntensity(masking_method='mask'))
        else:    
            transforms.append(tio.transforms.RescaleIntensity())
    if args.normalization:
        if args.masking:
            transforms.append(tio.transforms.ZNormalization(masking_method='mask'))
        else:    
            transforms.append(tio.transforms.ZNormalization())
    if args.masking:    
        transforms.append(tio.transforms.Mask(masking_method='mask'))
    if args.resample > 0:
        transforms.append(tio.transforms.Resample(args.resample,image_interpolation = 'bspline')) # or linear for zero background
    if args.resize > 0:
        transforms.append(tio.transforms.Resize(args.resize,image_interpolation = 'bspline'))
    if args.croporpad > 0:
        transforms.append(tio.transforms.CropOrPad(target_shape=args.croporpad))
    if args.canonical:
        transforms.append(tio.transforms.ToCanonical())
                          
    composed_transforms = tio.Compose(transforms)
    output_subject = composed_transforms(subject)

    output_subject.image.save(args.output_image)    