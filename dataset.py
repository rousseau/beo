from utils import load_images, nbimages_to_4darrays, array_normalization, nd_windowing
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state
from scipy import signal
import matplotlib.pyplot as plt

from os.path import expanduser

def get_ixi_4darrays():
  home = expanduser("~")
  ixi_path = home+'/Exp/IXImedium/'
  T1s = nbimages_to_4darrays(load_images(ixi_path, key='*T1.nii.gz', loader='nb',verbose=1))
  T2s = nbimages_to_4darrays(load_images(ixi_path, key='*T2.nii.gz', loader='nb',verbose=1))
  PDs = nbimages_to_4darrays(load_images(ixi_path, key='*PD.nii.gz', loader='nb',verbose=1))
  masks = nbimages_to_4darrays(load_images(ixi_path, key='*mask.nii.gz', loader='nb',verbose=1))
  return (T1s,T2s,PDs,masks)

def get_ixi_3dpatches(patch_size = 40, n_patches = 1000):

  (T1s,T2s,PDs,masks) = get_ixi_4darrays()  
  n_images = len(T1s)
  
  T1_patches = None
  T2_patches = None
  PD_patches = None
  
  mask_extract = 0.5
  
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
    #print(patches.shape)
    
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
          
#  plt.figure()
#  plt.subplot(1,3,1)
#  plt.imshow(T1_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.subplot(1,3,2)
#  plt.imshow(T2_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.subplot(1,3,3)
#  plt.imshow(PD_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.show(block=False)
  
#  for n in range(T1_patches.shape[0]):
#    T1_patches[n,:,:,:] = nd_windowing(T1_patches[n,:,:,:],signal.hamming)
#    T2_patches[n,:,:,:] = nd_windowing(T2_patches[n,:,:,:],signal.hamming)
#    PD_patches[n,:,:,:] = nd_windowing(PD_patches[n,:,:,:],signal.hamming)
  
#  plt.figure()
#  plt.subplot(1,3,1)
#  plt.imshow(T1_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.subplot(1,3,2)
#  plt.imshow(T2_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.subplot(1,3,3)
#  plt.imshow(PD_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.show(block=False)
  
  return (T1_patches,T2_patches,PD_patches)

def get_ixi_2dpatches(patch_size = 40, n_patches = 1000):
  
  (T1s,T2s,PDs,masks) = get_ixi_4darrays()  
  n_images = len(T1s)
  
  T1_patches = None
  T2_patches = None
  PD_patches = None
  
  mask_extract = 0.5
  patch_shape = (patch_size,patch_size,1)
  random_state = None

  for i in tqdm(range(n_images)):
    #Normalize data using mask
    T1_norm = array_normalization(X=T1s[i],M=masks[i],norm=0)
    T2_norm = array_normalization(X=T2s[i],M=masks[i],norm=0)
    PD_norm = array_normalization(X=PDs[i],M=masks[i],norm=0)
    mask = masks[i]
    
    for j in range(T1_norm.shape[2]): #Loop over the slices
      pT1 = extract_patches(T1_norm[:,:,j,:], patch_shape, extraction_step = 1)
      pT2 = extract_patches(T2_norm[:,:,j,:], patch_shape, extraction_step = 1)
      pPD = extract_patches(PD_norm[:,:,j,:], patch_shape, extraction_step = 1)
      pmask = extract_patches(mask[:,:,j,:], patch_shape, extraction_step = 1)
            
      rng = check_random_state(random_state)
      i_s = rng.randint(T1_norm.shape[0] - patch_shape[0] + 1, size = n_patches)
      j_s = rng.randint(T1_norm.shape[1] - patch_shape[1] + 1, size = n_patches)
      
      pT1 = pT1[i_s, j_s,:]
      pT2 = pT2[i_s, j_s,:]
      pPD = pPD[i_s, j_s,:]
      pmask = pmask[i_s, j_s,:]
      
      #Channel last
      pT1 = pT1.reshape(-1, patch_shape[0], patch_shape[1],patch_shape[2])
      pT2 = pT2.reshape(-1, patch_shape[0], patch_shape[1],patch_shape[2])
      pPD = pPD.reshape(-1, patch_shape[0], patch_shape[1],patch_shape[2])
      pmask = pmask.reshape(-1, patch_shape[0], patch_shape[1],patch_shape[2])
      
      #Remove empty patches (<50% of brain mask)  
      pmT1 = pT1[ np.mean(pmask,axis=1)>=mask_extract ]
      pmT2 = pT2[ np.mean(pmask,axis=1)>=mask_extract ]
      pmPD = pPD[ np.mean(pmask,axis=1)>=mask_extract ]  
      
      if T1_patches is None:
        T1_patches = np.copy(pmT1)
        T2_patches = np.copy(pmT2)
        PD_patches = np.copy(pmPD)    
      else:
        T1_patches = np.concatenate((T1_patches,pmT1),axis=0)
        T2_patches = np.concatenate((T2_patches,pmT2),axis=0)    
        PD_patches = np.concatenate((PD_patches,pmPD),axis=0)    
                
  print(T1_patches.shape)      
  plt.figure()
  plt.subplot(1,3,1)
  plt.imshow(T1_patches[10,:,:,0],cmap='gray',vmin=-3,vmax=3)
  plt.subplot(1,3,2)
  plt.imshow(T2_patches[10,:,:,0],cmap='gray',vmin=-3,vmax=3)
  plt.subplot(1,3,3)
  plt.imshow(PD_patches[10,:,:,0],cmap='gray',vmin=-3,vmax=3)
  plt.show(block=False)
  
#  for n in range(T1_patches.shape[0]):
#    T1_patches[n,:,:,:] = nd_windowing(T1_patches[n,:,:,:],signal.hamming)
#    T2_patches[n,:,:,:] = nd_windowing(T2_patches[n,:,:,:],signal.hamming)
#    PD_patches[n,:,:,:] = nd_windowing(PD_patches[n,:,:,:],signal.hamming)
  
#  plt.figure()
#  plt.subplot(1,3,1)
#  plt.imshow(T1_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.subplot(1,3,2)
#  plt.imshow(T2_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.subplot(1,3,3)
#  plt.imshow(PD_patches[10,:,:,20],cmap='gray',vmin=-3,vmax=3)
#  plt.show(block=False)
  
  return (T1_patches,T2_patches,PD_patches)

