#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchio as tio
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Beo data normalization')
  parser.add_argument('-i', '--input', help='Input image (nifti)', type=str, required=True)
  parser.add_argument('-o', '--output', help='Output image (nifti)', type=str, required=True)
  parser.add_argument('-n', '--norm', help='Normalization method', type=str, required=False, default='znorm')

  args = parser.parse_args()

  subject = tio.Subject(
    image=tio.ScalarImage(args.input),
  )

  normalization = tio.ZNormalization()
  transformed_subject = normalization(subject)
  transformed_subject.image.save(args.output)