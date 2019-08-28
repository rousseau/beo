import glob
import nibabel
import SimpleITK as sitk
import numpy as np


def load_images(data_path, key, loader = 'itk', verbose = 0):  
  """Load image data in a specified directory using a key word"""  
  """Data can be loaded using itk or nibabel"""
  if verbose !=0:
    print('loading images')
  images = []
  #load all the images
  directories = glob.glob(data_path, recursive=True)
  if verbose !=0:
    print(directories)
  all_files = []
  for d in directories:
    if verbose !=0:
      print('Looking for data in '+d)
    files = glob.glob(d+key)
    all_files.extend(files)
  all_files.sort()
  
  for file in all_files:  
    if verbose !=0:
      print('Loading : '+file)
    if loader == 'itk':
      #Load images using SimpleITK
      images.append(sitk.ReadImage(file))  
    else:      
      #Load images using nibabel 
      images.append(nibabel.load(file))
    
  return images

def nbimages_to_4darrays(images):
  #Convert a list of nibabel images into a list of 4D array
  arrays = []
  for im in images:
    data  = np.float32(im.get_data())
    #Make sure that all data are 4D
    if len(data.shape)==3:
      data = data[:,:,:,np.newaxis]
    arrays.append(data)  
  return arrays

def array_normalization(X,M=None,norm=0):
  """Normalization for image regression
    Inputs : 
      X : array of 4D data
      M : array of 4D mask used to compute normalization parameters
    Outputs :
      X : normalized data
  """
  if M is None:
    M = np.ones(X.shape)
    
  #normalization using the ROI defined by the mask
        
  if norm == 0:
    #Zero-centered
    X = (X - np.mean(X[M==1])) / np.std(X[M==1])
  else:  
    #[0,1] normalization
    maxX = np.max(X[M==1])
    minX = np.min(X[M==1])
    X = (X - minX)/(maxX-minX)

  return X