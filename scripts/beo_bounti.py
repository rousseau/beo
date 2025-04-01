#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from beo_wrappers import wrapper_svrtk_segmentation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply BOUNTI on a single file')
    parser.add_argument('-i', '--input', help='Input file (absolute path)', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output file (absolute path)', type=str, required=True)

    args = parser.parse_args()
 
    wrapper_svrtk_segmentation(args.input, args.output)
