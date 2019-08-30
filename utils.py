import glob
import nibabel
import SimpleITK as sitk
import numpy as np
import keras.backend as K


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
    #Make sure that all data are 4D (channels last)
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

#Windowing to avoid boundary effect when applying convolutions
#nd_windowing function from https://stackoverflow.com/questions/27345861/extending-1d-function-across-3-dimensions-for-data-windowing
def nd_windowing(data, filter_function):
  """
  Performs an in-place windowing on N-dimensional spatial-domain data.
  This is done to mitigate boundary effects in the FFT.
  
  Parameters
  ----------
  data : ndarray
         Input data to be windowed, modified in place.
  filter_function : 1D window generation function
         Function should accept one argument: the window length.
         Example: scipy.signal.hamming
  """
  
  for axis, axis_size in enumerate(data.shape):
    # set up shape for numpy broadcasting
    filter_shape = [1, ] * data.ndim
    filter_shape[axis] = axis_size
    window = filter_function(axis_size).reshape(filter_shape)
    # scale the window intensities to maintain image intensity
    #np.power(window, (1.0/data.ndim), out=window)  
    data *= window
  return data    

def apply_model_on_3dimage(model,image,mask=None):
  """
  Apply a Keras model on a 3D nibabel image.
  A mask (numpy array) can be used for normalization
  """
  array = image.get_data().astype(float)
  #Convert 3D array to 4D array with channel last
  if len(array.shape) < 4:
    array = np.expand_dims(array,axis=-1) #by default we use channel last
  
  if mask is not None:
    if len(mask.shape) < 4:
      mask = np.expand_dims(mask,axis=-1)
  else:
    mask = np.ones(array.shape)
    
  array = array_normalization(X=array,M=mask,norm=0)
  #Add fifth dimension for Keras prediction
  array = np.expand_dims(array,axis=0)
  
  if K.image_data_format() == 'channels_first':
    print('Swapping axis because channel first mode is used in Keras')
    array = np.moveaxis(array, -1, 1)  
    
  output_array = model.predict(array, batch_size=1)
  
  if K.image_data_format() == 'channels_first':
    output_image = nibabel.Nifti1Image(np.squeeze(output_array[0,0,:,:,:]), image.affine)
  else:
    output_image = nibabel.Nifti1Image(np.squeeze(output_array[0,:,:,:,0]), image.affine)  
  
  return output_image
  
  
  