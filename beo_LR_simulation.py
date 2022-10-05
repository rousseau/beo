# -*- coding: utf-8 -*-

import argparse



# Objective: simulation LR image using a given PSF
# Inputs: a reference (HR) image and a LR image (created using ITK-based resampling)
# We use ITK resampling because it's a simple way to obtain the new pixel coordinates of LR image 
# Otherwise, we have to compute new coordinates depending on image resolutions (HR and LR)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-r', '--ref',   help='Reference Image filename (i.e. ground truth) (required)', type=str, required = True)
  parser.add_argument('-i', '--input', help='Low-resolution image filename (required), created using ITK-based Resampling', type=str, required = True)
  parser.add_argument('-o', '--output', help='Low-resolution simulated image filename (required)', type=str, required = True)
