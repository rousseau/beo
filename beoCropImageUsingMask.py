import argparse
import SimpleITK as sitk
import numpy as np

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('-i', '--input', help='Input Image', type=str, required=True) 
  parser.add_argument('-m', '--mask', help='Input Mask', type=str, required=False)  
  parser.add_argument('-o', '--output', help='Output Image', type=str, required=True)
  parser.add_argument('-r', '--radius', help='Mask dilation radius', type=int, default=0, required=False)
  args = parser.parse_args()

  #Read input image
  img = sitk.ReadImage( args.input )
  pixelID = img.GetPixelID()

  #Read mask image
  mask = sitk.ReadImage( args.mask )

  #Dilate mask if needed
  if args.radius > 0:
    mask = sitk.BinaryDilate(mask, args.radius)

  #Define ROI from mask values
  nda = sitk.GetArrayFromImage(mask)

  a = np.where(nda != 0)
  size = [int(np.max(a[2])-np.min(a[2])),int(np.max(a[1])-np.min(a[1])),int(np.max(a[0])-np.min(a[0]))]
  index = (int(np.min(a[2])),int(np.min(a[1])),int(np.min(a[0])))
  print(size)
  print(index)
  
  roi_img = sitk.RegionOfInterest(img,size,index)

  #Write ouput image
  sitk.WriteImage( sitk.Cast( roi_img, pixelID ), args.output )
