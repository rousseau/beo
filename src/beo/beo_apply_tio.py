#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse

import torchio as tio
import numpy

#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply torchio transforms on nifti images.')
    parser.add_argument('-i', '--input_image', help='Input nifti image ', type=str, required = True)
    parser.add_argument('-o', '--output_image', help='Output nifti image ', type=str, required = True)

    parser.add_argument('-m', '--mask', help='Mask nifti image ', type=str, required = False)
    parser.add_argument('--output_mask', help='Output nifti mask image ', type=str, required = False)

    parser.add_argument('-l', '--label', help='Label nifti image ', type=str, required = False)

    parser.add_argument('--rescale', help='Rescale intensity', action='store_true')
    parser.add_argument('--normalization', help='Z normalization', action='store_true')
    parser.add_argument('--masking', help='Apply Mask', action='store_true')
    parser.add_argument('--croporpad', help='Image size for Crop or pad', type=int, required=False, default=0)
    parser.add_argument('--canonical', help='To canonical', action='store_true')
    parser.add_argument('--resample', help='Voxel size for Resample', type=float, required=False, default=0)
    parser.add_argument('--resize', help='Image size for Resize', type=float, required=False, default=0)

    parser.add_argument('--onehot', help='Apply one hot encoding', action='store_true')

    parser.add_argument('--interp', help='Interpolation mode (bspline, linear, nearest)', type=str, required=False, default='bspline')

    args = parser.parse_args()
    print(args)

    sub_dict = {}
    arg_dict = {}
    
    if args.onehot:
        sub_dict['image'] = tio.LabelMap(args.input_image)
    else:
        sub_dict['image'] = tio.ScalarImage(args.input_image)

    if args.mask:
        sub_dict['mask'] = tio.LabelMap(args.mask)
    if args.label:
        sub_dict['label'] = tio.LabelMap(args.label)    


    subject = tio.Subject(sub_dict)

    transforms = []

    if args.rescale:
        transforms.append(tio.transforms.RescaleIntensity(percentiles=(0.1, 99.9)))
        transforms.append(tio.transforms.Clamp(out_min=0, out_max=1))

    if args.normalization:
        if args.masking:
            transforms.append(tio.transforms.ZNormalization(masking_method='mask'))
        else:    
            transforms.append(tio.transforms.ZNormalization())

    if args.masking:    
        transforms.append(tio.transforms.Mask(masking_method='mask'))
    if args.onehot:
        transforms.append(tio.transforms.OneHot())

    if args.resample > 0:
        transforms.append(tio.transforms.Resample(args.resample,image_interpolation = args.interp)) # or linear for zero background
    if args.resize > 0:
        transforms.append(tio.transforms.Resize(args.resize,image_interpolation = args.interp))
    if args.croporpad > 0:
        transforms.append(tio.transforms.CropOrPad(target_shape=args.croporpad))
    if args.canonical:
        transforms.append(tio.transforms.ToCanonical())
                          
    print(transforms)
    composed_transforms = tio.Compose(transforms)
    output_subject = composed_transforms(subject)


    o = tio.ScalarImage(tensor=output_subject.image.data.detach().numpy().astype(numpy.float32), affine=output_subject.image.affine)
    o.save(args.output_image)
    #output_subject.image.save(args.output_image) 
       
    if args.output_mask is not None:
        output_subject.mask.save(args.output_mask)    