# -*- coding: utf-8 -*-

import argparse
import numpy as np
import nibabel

# Objective: simulation LR image using a given PSF
# Inputs: a reference (HR) image and a LR image (created using ITK-based resampling)
# We use ITK resampling because it's a simple way to obtain the new pixel coordinates of LR image 
# Otherwise, we have to compute new coordinates depending on image resolutions (HR and LR)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-r', '--ref',   help='Reference Image filename (i.e. ground truth) (required)', type=str, required = True)
  parser.add_argument('-i', '--input', help='Low-resolution image filename (required), created using ITK-based Resampling', type=str, required = True)
  parser.add_argument('-o', '--output', help='Low-resolution simulated image filename (required)', type=str, required = True)

  args = parser.parse_args()

  #Load images
  HRimage = nibabel.load(args.ref)
  LRimage = nibabel.load(args.input)

  #Get image resolution
  HRSpacing = np.float32(np.array(HRimage.header['pixdim'][1:4]))  
  LRSpacing = np.float32(np.array(LRimage.header['pixdim'][1:4]))  

  #use a identity transform for now
  inputTransform = None
  #no transform provided : use identity as transform and zero as center
  m = np.identity(4)
  c = np.array([0, 0, 0, 1])
  inputTransform = (m,c) 

  #Pre-compute PSF values
  #PSF is a box centered around an observed pixel of LR image
  #The size of the box is set as the size of a LR pixel (expressed in voxel space)
  n_samples = 5
  psf_sx = np.linspace(-0.5,0.5,n_samples)
  psf_sy = np.linspace(-0.5,0.5,n_samples)
  psf_sz = np.linspace(-0.5,0.5,n_samples)

  #Define a set of points for PSF values using meshgrid
  #https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
  psf_x, psf_y, psf_z = np.meshgrid(psf_sx, psf_sy, psf_sz, indexing='ij')

  #Define gaussian kernel as PSF model
  sigma = 1.0 / 2.3548 #could be anisotropic to reflect MRI sequences (see Kainz et al.)
  def gaussian(x,sigma):
    return np.exp(-x*x/(2*sigma*sigma))  

  psf = gaussian(psf_x,sigma) * gaussian(psf_y, sigma) * gaussian(psf_z,sigma) 
  psf = psf / np.sum(psf)

  nibabel.save(nibabel.Nifti1Image(psf, HRimage.affine),'toto.nii.gz')    

