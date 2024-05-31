#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl

import torchio as tio
import math
import numpy as np



#%% Main program
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo Registration 3D Image Pair')
    parser.add_argument('-t', '--target', help='Target / Reference Image', type=str, required = True)
    parser.add_argument('-s', '--source', help='Source / Moving Image', type=str, required = True)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, required = False, default=1)
    parser.add_argument('-o', '--output', help='Output filename', type=str, required = True)

    args = parser.parse_args()

    subject = tio.Subject(
        target=tio.ScalarImage(args.target),
        source=tio.ScalarImage(args.source),
    )
    # image shape XYZ


    batch_size = 1

    scale = torch.Tensor([1,1,1])
    shear = torch.Tensor([0,0,0])
    euler_angles_deg = torch.Tensor([0,0,0])
    translation = torch.Tensor([0,0,0])

    # Make it in batch mode
    scale = torch.unsqueeze(scale,0)
    shear = torch.unsqueeze(shear,0)
    euler_angles_deg = torch.unsqueeze(euler_angles_deg,0)
    translation = torch.unsqueeze(translation,0)

    # Build scale matrix
    scale_matrix = torch.unsqueeze(torch.eye(3),0)
    scale_matrix = scale_matrix.repeat(batch_size, 1, 1)
    scale_matrix[:,0,0] = scale[:, 0]
    scale_matrix[:,1,1] = scale[:, 1]
    scale_matrix[:,2,2] = scale[:, 2]

    # Build shear matrix
    shear_matrix = torch.unsqueeze(torch.eye(3),0)
    shear_matrix = shear_matrix.repeat(batch_size, 1, 1)
    shear_matrix[:,0,1] = shear[:, 0]
    shear_matrix[:,1,0] = shear[:, 1]
    shear_matrix[:,0,2] = shear[:, 2]

    # Convert Euler angles to rotation matrices (assuming ZYX convention)
    euler_angles = euler_angles_deg * math.pi / 180.0
    cos_x = torch.cos(euler_angles[:, 0])
    sin_x = torch.sin(euler_angles[:, 0])
    cos_y = torch.cos(euler_angles[:, 1])
    sin_y = torch.sin(euler_angles[:, 1])
    cos_z = torch.cos(euler_angles[:, 2])
    sin_z = torch.sin(euler_angles[:, 2])

    rotation_x = torch.stack([
        torch.ones_like(cos_x), torch.zeros_like(cos_x), torch.zeros_like(cos_x),
        torch.zeros_like(cos_x), cos_x, -sin_x,
        torch.zeros_like(cos_x), sin_x, cos_x
    ], dim=-1).view(batch_size, 3, 3)

    rotation_y = torch.stack([
        cos_y, torch.zeros_like(cos_y), -sin_y,
        torch.zeros_like(cos_y), torch.ones_like(cos_y), torch.zeros_like(cos_y),
        sin_y, torch.zeros_like(cos_y), cos_y
    ], dim=-1).view(batch_size, 3, 3)

    rotation_z = torch.stack([
        cos_z, -sin_z, torch.zeros_like(cos_z),
        sin_z, cos_z, torch.zeros_like(cos_z),
        torch.zeros_like(cos_z), torch.zeros_like(cos_z), torch.ones_like(cos_z)
    ], dim=-1).view(batch_size, 3, 3)

    # Combine rotations and build the affine transform matrix
    rotation_matrix = torch.matmul(torch.matmul(rotation_x, rotation_y), rotation_z)

    print(rotation_matrix.shape)
    print(shear_matrix.shape)
    print(scale_matrix.shape)

    scaled_shear_matrix = torch.bmm(scale_matrix, torch.bmm(shear_matrix, rotation_matrix))
    affine_matrix = F.pad(scaled_shear_matrix, pad=(0, 1, 0, 1), value=0.0)
    affine_matrix[:, 3, :3] = translation
    affine_matrix[:,3,3] = 1

    print(affine_matrix)
    print(affine_matrix.shape)

    # Create displacement field (affine_grid)
    

    # Apply to source image