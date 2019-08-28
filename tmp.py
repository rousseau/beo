from utils import load_images, nbimages_to_4darrays, array_normalization
from tqdm import tqdm
import keras.backend as K
import numpy as np
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state

print(K.image_data_format())
#DATA FORMAT : channel last

ixi_path = '/home/rousseau/Exp/IXIsmall/'
T1s = nbimages_to_4darrays(load_images(ixi_path, key='*T1.nii.gz', loader='nb',verbose=1))
T2s = nbimages_to_4darrays(load_images(ixi_path, key='*T2.nii.gz', loader='nb',verbose=1))
PDs = nbimages_to_4darrays(load_images(ixi_path, key='*PD.nii.gz', loader='nb',verbose=1))
masks = nbimages_to_4darrays(load_images(ixi_path, key='*mask.nii.gz', loader='nb',verbose=1))
n_images = len(T1s)

T1_patches = None
T2_patches = None
PD_patches = None

mask_extract = 0.5
n_patches = 1000
patch_size = 40

for i in tqdm(range(n_images)):
  #Normalize data using mask
  T1_norm = array_normalization(X=T1s[i],M=masks[i],norm=0)
  T2_norm = array_normalization(X=T2s[i],M=masks[i],norm=0)
  PD_norm = array_normalization(X=PDs[i],M=masks[i],norm=0)
  mask = masks[i]
  
  #Concatenate all modalities into a 4D volume
  vol4d = np.concatenate((T1_norm, T2_norm, PD_norm, mask), axis=3)

  #Extract random 4D patches
  random_state = None
  extraction_step = 1
  patch_shape = (patch_size,patch_size,patch_size,4)
  patches = extract_patches(vol4d, patch_shape, extraction_step=extraction_step)
  
  rng = check_random_state(random_state)
  i_s = rng.randint(vol4d.shape[0] - patch_shape[0] + 1, size=n_patches)
  j_s = rng.randint(vol4d.shape[1] - patch_shape[1] + 1, size=n_patches)
  k_s = rng.randint(vol4d.shape[2] - patch_shape[2] + 1, size=n_patches)
  
  patches = patches[i_s, j_s, k_s, :]
  patches = patches.reshape(-1, patch_shape[0],patch_shape[1],patch_shape[2], 4)
  print(patches.shape)
  
  pT1= patches[:,:,:,:,0]
  pT2= patches[:,:,:,:,1]  
  pPD= patches[:,:,:,:,2]  
  pM = patches[:,:,:,:,3]
  pM = pM.reshape(pM.shape[0],-1)  
  
  #Remove empty patches (<50% of brain mask)  
  pmT1 = pT1[ np.mean(pM,axis=1)>=mask_extract ]
  pmT2 = pT2[ np.mean(pM,axis=1)>=mask_extract ]
  pmPD = pPD[ np.mean(pM,axis=1)>=mask_extract ]  
  
  if T1_patches is None:
    T1_patches = np.copy(pmT1)
    T2_patches = np.copy(pmT2)
    PD_patches = np.copy(pmPD)    
  else:
    T1_patches = np.concatenate((T1_patches,pmT1),axis=0)
    T2_patches = np.concatenate((T2_patches,pmT2),axis=0)    
    PD_patches = np.concatenate((PD_patches,pmPD),axis=0)    
    
print(T1_patches.shape)    