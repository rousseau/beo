import os

import numpy as np
from scipy.ndimage import distance_transform_cdt, gaussian_filter
import argparse
import nibabel


def sdf(x, sigma=1.0, normalize=1):
    sdf_in = distance_transform_cdt(x)
    sdf_out = distance_transform_cdt(1-x)
    sdf = (sdf_out-sdf_in)*1.0 #/ np.max(np.abs(sdf_in))
    if normalize==1:
        sdf = sdf / np.max(np.abs(sdf))
    if sigma > 0:
        sdf = gaussian_filter(sdf,sigma=sigma)
    #sdf = np.clip(sdf,-20,20)
    return sdf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute signed distance field from binary mask')
    parser.add_argument('-i', '--input', help='Input Nifti image', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output Nifti image', type=str, required=True)
    parser.add_argument('-l', '--label', help='List of label to use', type=str, nargs='+', required=True)
    parser.add_argument('-s', '--sigma', help='Sigma of Gaussian filtering', type=float, required=False, default=0.0)
    parser.add_argument('-n', '--normalization', help='Normalization [0,1]', type=int, required=False, default=1)

    args = parser.parse_args()

    seg = nibabel.load(args.input)
    seg_data = seg.get_fdata()

    bin_seg = np.zeros(seg_data.shape)
    for l in args.label:
        bin_seg += np.where(seg_data == int(l), 1, 0)

    nibabel.save( nibabel.Nifti1Image(sdf(bin_seg,sigma=args.sigma,normalize=args.normalization).astype(np.float32), seg.affine), args.output)
