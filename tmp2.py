from dataset import get_ixi_3dpatches
from model_zoo import ResNet3D, AutoEncoder3D, CHP, TriangleModel
from model_zoo import build_feature_model, build_recon_model, build_block_model
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition
from model_zoo import build_exp_model, fm_model
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import nibabel
from utils import apply_model_on_3dimage, freeze_model
from keras.models import load_model
import joblib
from sklearn.utils import check_random_state

#Package from OpenAI for GPU memory saving
#https://github.com/cybertronai/gradient-checkpointing
import memory_saving_gradients

print(K.image_data_format())
from os.path import expanduser
home = expanduser("~")
output_path = home+'/Sync/Experiments/IXI/'

patch_size = 40

load_pickle_patches = 1

if load_pickle_patches == 0:
  n_patches = 2500
  (T1,T2,PD) = get_ixi_3dpatches(patch_size = patch_size, n_patches = n_patches)
  
  joblib.dump(T1,home+'/Exp/T1.pkl', compress=True)
  joblib.dump(T2,home+'/Exp/T2.pkl', compress=True)
  joblib.dump(PD,home+'/Exp/PD.pkl', compress=True)
  
else:
  print('Loading gzip pickle files')  
  T1 = joblib.load(home+'/Exp/T1.pkl')
  T2 = joblib.load(home+'/Exp/T2.pkl')
  PD = joblib.load(home+'/Exp/PD.pkl')

if K.image_data_format() == 'channels_first':
  T1 = np.expand_dims(T1,axis=1)
  T2 = np.expand_dims(T2,axis=1)
  PD = np.expand_dims(PD,axis=1)  
else:
  T1 = np.expand_dims(T1,axis=-1)
  T2 = np.expand_dims(T2,axis=-1)
  PD = np.expand_dims(PD,axis=-1)  

print(T1.shape)

n_channelsX = 1
n_channelsY = 1
n_filters = 32
n_layers = 1
n_layers_residual = 5
learning_rate = 0.0001
loss = 'mae' 
batch_size = 32 
epochs = 20
use_optim = 0

inverse_consistency = 0
cycle_consistency = 0
identity_consistency = 0


prefix = 'iclr_nowindowing'
if use_optim == 1:
  prefix += '_optim'
if inverse_consistency == 1:
  prefix += '_invc'
if cycle_consistency == 1:
  prefix += '_cycle'
if identity_consistency == 1:
  prefix += '_idc'  
prefix+= '_e'+str(epochs)+'_ps'+str(patch_size)+'_np'+str(T1.shape[0])
prefix+= '_bs'+str(batch_size)
prefix+= '_lr'+str(learning_rate)
prefix+= '_nl'+str(n_layers)
prefix+= '_nlr'+str(n_layers_residual)
prefix+= '_'

if K.image_data_format() == 'channels_first':
  init_shape = (n_channelsX, None, None, None)
  feature_shape = (n_filters, None, None, None)
  channel_axis = 1
else:
  init_shape = (None, None, None,n_channelsX)
  feature_shape = (None, None, None, n_filters)
  channel_axis = -1

#The models required to build all the models :   
feature_model = build_feature_model(init_shape=init_shape, n_filters=n_filters, kernel_size=5)
recon_model = build_recon_model(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=3)

block_T2_to_T1 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual) 
  

#Identity consistency
model_identity = build_one_model(init_shape, feature_model = feature_model, mapping_model = None, reconstruction_model = recon_model)
model_identity.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
feature_model.summary()
recon_model.summary()


#Direct mapping
mapping_T2_to_T1 = build_forward_model(init_shape=feature_shape, block_model=block_T2_to_T1, n_layers=n_layers)
model_T2_to_T1 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_T2_to_T1, reconstruction_model = recon_model)
model_T2_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 


def subsample(sampling=0.1):
  random_state = None
  rng = check_random_state(random_state)
  if sampling <= 1:
    n_samples = int(T1.shape[0]*sampling)
  else:
    n_samples = int(sampling)
  index = rng.randint(T1.shape[0], size=n_samples)
  return T1[index,:,:,:,:],T2[index,:,:,:,:],PD[index,:,:,:,:]

for e in range(epochs):
  #Training on small batch size
  for i in range(int(T1.shape[0]/batch_size)):
    subT1, subT2, subPD = subsample(batch_size)
    model_T2_to_T1.fit(x=subT2, y=subT1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
    
    freeze_model(feature_model,freeze=1)
    model_identity.fit(x=subT2, y=subT2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)   
    model_identity.fit(x=subT1, y=subT1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
    freeze_model(feature_model,freeze=0)    
    
    
    
  

prefix = 'experimental_'
   
#Apply on a test image
T1image = nibabel.load(output_path+'IXI661-HH-2788-T1_N4.nii.gz')
T2image = nibabel.load(output_path+'IXI661-HH-2788-T2_N4.nii.gz')
PDimage = nibabel.load(output_path+'IXI661-HH-2788-PD_N4.nii.gz')
maskarray = nibabel.load(output_path+'IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)




feature_model.save(output_path+prefix+'feature_model.h5')
recon_model.save(output_path+prefix+'recon_model.h5')
block_PD_to_T2.save(output_path+prefix+'block_pd_to_t2.h5')
block_T2_to_T1.save(output_path+prefix+'block_t2_to_t1.h5')
block_T1_to_PD.save(output_path+prefix+'block_t1_to_pd.h5')


#Encoding
image_T1_id = apply_model_on_3dimage(model_identity, T1image, mask=maskarray)
nibabel.save(image_T1_id,output_path+prefix+'id_T1.nii.gz')

image_T2_id = apply_model_on_3dimage(model_identity, T2image, mask=maskarray)
nibabel.save(image_T2_id,output_path+prefix+'id_T2.nii.gz')

#Direct mapping

image_T2_to_T1 = apply_model_on_3dimage(model_T2_to_T1, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,output_path+prefix+'direct_T2_to_T1.nii.gz')

    


