from dataset import get_ixi_3dpatches
from model_zoo import ResNet3D, AutoEncoder3D, CHP, TriangleModel
from model_zoo import build_feature_model, build_recon_model, build_block_model
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition
from model_zoo import build_exp_model, fm_model, build_reversible_forward_model, build_reversible_backward_model, build_4_model
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import nibabel
from utils import apply_model_on_3dimage
from keras.models import load_model
import joblib
from sklearn.utils import check_random_state
from keras.utils.generic_utils import Progbar

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
n_filters = 16
n_layers = 1
n_layers_residual = 5
learning_rate = 0.001
loss = 'mae' 
batch_size = 32 
epochs = 50
use_optim = 0
kernel_size = 5

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
  half_feature_shape = ((int)(n_filters/2), None, None, None)
  channel_axis = 1
else:
  init_shape = (None, None, None,n_channelsX)
  feature_shape = (None, None, None, n_filters)
  half_feature_shape = (None, None, None, (int)(n_filters/2))
  channel_axis = -1

#The models required to build all the models :   
feature_model = build_feature_model(init_shape=init_shape, n_filters=n_filters, kernel_size=kernel_size)
recon_model = build_recon_model(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=kernel_size)

block_PD_to_T2 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual) 
block_T2_to_T1 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual) 
block_T1_to_PD =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=n_layers_residual) 
  

block_f_T2_to_T1 = build_block_model(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)
block_g_T2_to_T1 = build_block_model(init_shape=half_feature_shape, n_filters=(int)(n_filters/2), n_layers=n_layers_residual)


#Identity consistency
model_identity = build_one_model(init_shape, feature_model = feature_model, mapping_model = None, reconstruction_model = recon_model)
#model_identity.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
feature_model.summary()
recon_model.summary()

#Direct mapping
mapping_PD_to_T2 = build_forward_model(init_shape=feature_shape, block_model=block_PD_to_T2, n_layers=n_layers)
mapping_T2_to_T1 = build_forward_model(init_shape=feature_shape, block_model=block_T2_to_T1, n_layers=n_layers)
mapping_T1_to_PD = build_forward_model(init_shape=feature_shape, block_model=block_T1_to_PD, n_layers=n_layers)

model_PD_to_T2 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_PD_to_T2, reconstruction_model = recon_model)
model_T2_to_T1 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_T2_to_T1, reconstruction_model = recon_model)
model_T1_to_PD = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_T1_to_PD, reconstruction_model = recon_model)

model_PD_to_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_T2_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_T1_to_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 


mapping_reversible_T2_to_T1 = build_reversible_forward_model(init_shape=feature_shape, block_f=block_f_T2_to_T1, block_g = block_g_T2_to_T1, n_layers=n_layers)
mapping_reversible_T1_to_T2 = build_reversible_backward_model(init_shape=feature_shape, block_f=block_f_T2_to_T1, block_g = block_g_T2_to_T1, n_layers=n_layers)

model_reversible_T2_to_T1 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_reversible_T2_to_T1, reconstruction_model = recon_model)
model_reversible_T1_to_T2 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_reversible_T1_to_T2, reconstruction_model = recon_model)

model_reversible_T2_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_reversible_T1_to_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 


model_exp = fm_model(init_shape, feature_model, mapping_reversible_T2_to_T1)

#For mapping model, r is not trainable
#feature_model.trainable = True    
#recon_model.trainable = False    
model_exp.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

#For identity model, f is not trainable
#feature_model.trainable = False    
#recon_model.trainable = True    
model_identity.compile(optimizer=Adam(lr=learning_rate), loss=loss)

model_warped = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_reversible_T2_to_T1, reconstruction_model = None)
model_warped.compile(optimizer=Adam(lr=learning_rate), loss=loss)

model_all = build_4_model(init_shape, feature_model, mapping_reversible_T2_to_T1, model_reversible_T1_to_T2, recon_model)
model_all.compile(optimizer=Adam(lr=learning_rate), 
                  loss=loss,
                  loss_weights=[1,1,1,1]) 

#models = []
#models.append(model_identity)
#models.append(model_T2_to_T1)
#models.append(model_exp)
#
#for m in models:
#  m.compile(optimizer=Adam(lr=learning_rate), loss=loss)   


def subsample(sampling=0.1):
  random_state = None
  rng = check_random_state(random_state)
  if sampling <= 1:
    n_samples = int(T1.shape[0]*sampling)
  else:
    n_samples = int(sampling)
  index = rng.randint(T1.shape[0], size=n_samples)
  return T1[index,:,:,:,:],T2[index,:,:,:,:],PD[index,:,:,:,:]

#Apply on a test image
T1image = nibabel.load(output_path+'IXI661-HH-2788-T1_N4.nii.gz')
T2image = nibabel.load(output_path+'IXI661-HH-2788-T2_N4.nii.gz')
PDimage = nibabel.load(output_path+'IXI661-HH-2788-PD_N4.nii.gz')
maskarray = nibabel.load(output_path+'IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)

random_state = None
rng = check_random_state(random_state)

#Start with mapping first? (most difficult)
#model_T2_to_T1.fit(x=T2, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)

for e in range(epochs):
   
  #Training on small batch size
  num_batches = int(np.ceil(T1.shape[0] / float(batch_size)))
  progress_bar = Progbar(target=num_batches)

  index = rng.randint(T1.shape[0], size=T1.shape[0])#tirage avec remise

  epoch_end_to_end_loss = []
  epoch_end_to_end_rev_loss = []
  epoch_mapping_loss = []
  epoch_idT2_loss = []
  epoch_idT1_loss = []
    
  for i in range(num_batches):
    #subT1, subT2, subPD = subsample(batch_size)
    subT1 = T1[index[i*batch_size:(i+1)*batch_size],:,:,:,:]
    subT2 = T2[index[i*batch_size:(i+1)*batch_size],:,:,:,:]
    subPD = PD[index[i*batch_size:(i+1)*batch_size],:,:,:,:]    
    
    #All
    epoch_end_to_end_loss.append(model_all.train_on_batch(x=[subT2,subT1], y=[subT1,subT2,subT2,subT1]))
    #Reversible
    #epoch_end_to_end_loss.append(model_reversible_T2_to_T1.train_on_batch(x=subT2, y=subT1))
    #epoch_end_to_end_rev_loss.append(model_reversible_T1_to_T2.train_on_batch(x=subT1, y=subT2))

    #if i%5 ==0:
    #model_reversible_T2_to_T1.train_on_batch(x=subT1, y=subT1)
    #  model_reversible_T1_to_T2.train_on_batch(x=subT2, y=subT2)
    
    
    #epoch_end_to_end_loss.append(model_T2_to_T1.train_on_batch(x=subT2, y=subT1))
    #model_PD_to_T2.train_on_batch(x=subT2, y=subT1)
    #model_T1_to_PD.train_on_batch(x=subT2, y=subT1)

    #model_T2_to_T1.train_on_batch(x=subT1, y=subT1)
    
    #epoch_mapping_loss.append(model_exp.train_on_batch(x=[subT2,subT1], y=np.zeros(subT2.shape[0]))) 
    
    #if i%5 ==0: 
    #  epoch_idT2_loss.append(model_identity.train_on_batch(x=subT2, y=subT2))   
    #  epoch_idT1_loss.append(model_identity.train_on_batch(x=subT1, y=subT1))
    #  model_identity.train_on_batch(x=subPD, y=subPD)      
    
    progress_bar.update(i + 1)
    
  end_to_end_loss = np.mean(np.array(epoch_end_to_end_loss), axis=0)
  end_to_end_rev_loss = np.mean(np.array(epoch_end_to_end_rev_loss), axis=0)
  mapping_loss = np.mean(np.array(epoch_mapping_loss), axis=0)
  idT2_loss = np.mean(np.array(epoch_idT2_loss), axis=0)
  idT1_loss = np.mean(np.array(epoch_idT1_loss), axis=0)
  print('End-to-end Loss :'+str(end_to_end_loss))
  print('End-to-end Rev Loss :'+str(end_to_end_rev_loss))
  print('Mapping Loss :'+str(mapping_loss))
  print('IdT2 Loss : '+str(idT2_loss))
  print('IdT1 Loss : '+str(idT1_loss))
  
  print('saving results')#-----------------------------------------------------
  image_T1_id = apply_model_on_3dimage(model_identity, T1image, mask=maskarray)
  nibabel.save(image_T1_id,output_path+prefix+'_current'+str(e)+'_id_T1.nii.gz')
  
  image_T2_id = apply_model_on_3dimage(model_identity, T2image, mask=maskarray)
  nibabel.save(image_T2_id,output_path+prefix+'_current'+str(e)+'_id_T2.nii.gz')

  image_T2_to_T1 = apply_model_on_3dimage(model_reversible_T2_to_T1, T2image, mask=maskarray)
  nibabel.save(image_T2_to_T1,output_path+prefix+'_current'+str(e)+'_direct_T2_to_T1.nii.gz')

  image_T1_to_T2 = apply_model_on_3dimage(model_reversible_T1_to_T2, T1image, mask=maskarray)
  nibabel.save(image_T1_to_T2,output_path+prefix+'_current'+str(e)+'_backward_T2_to_T1.nii.gz')

  image_T2_to_T1 = apply_model_on_3dimage(model_warped, T2image, mask=maskarray)
  nibabel.save(image_T2_to_T1,output_path+prefix+'_current'+str(e)+'warped_T2_to_T1.nii.gz')

  image_T1_f = apply_model_on_3dimage(feature_model, T1image, mask=maskarray)
  nibabel.save(image_T1_f,output_path+prefix+'_current'+str(e)+'feature_T1.nii.gz')

  image_T2_f = apply_model_on_3dimage(feature_model, T2image, mask=maskarray)
  nibabel.save(image_T2_f,output_path+prefix+'_current'+str(e)+'feature_T2.nii.gz') 


prefix = 'experimental_'
   

feature_model.save(output_path+prefix+'feature_model.h5')
recon_model.save(output_path+prefix+'recon_model.h5')
block_T2_to_T1.save(output_path+prefix+'block_t2_to_t1.h5')

#Encoding
image_T1_id = apply_model_on_3dimage(model_identity, T1image, mask=maskarray)
nibabel.save(image_T1_id,output_path+prefix+'id_T1.nii.gz')

image_T2_id = apply_model_on_3dimage(model_identity, T2image, mask=maskarray)
nibabel.save(image_T2_id,output_path+prefix+'id_T2.nii.gz')

#Direct mapping
image_T2_to_T1 = apply_model_on_3dimage(model_reversible_T2_to_T1, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,output_path+prefix+'direct_T2_to_T1.nii.gz')

#In feature space
image_T1_f = apply_model_on_3dimage(feature_model, T1image, mask=maskarray)
nibabel.save(image_T1_f,output_path+prefix+'feature_T1.nii.gz')

image_T2_f = apply_model_on_3dimage(feature_model, T2image, mask=maskarray)
nibabel.save(image_T2_f,output_path+prefix+'feature_T2.nii.gz')    

#feature_model = load_model(output_path+prefix+'feature_model.h5')
#recon_model = load_model(output_path+prefix+'recon_model.h5')
#block_T2_to_T1 = load_model(output_path+prefix+'block_t2_to_t1.h5')
#mapping_T2_to_T1 = build_forward_model(init_shape=feature_shape, block_model=block_T2_to_T1, n_layers=n_layers)

model_warped = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_T2_to_T1, reconstruction_model = None)
image_T2_to_T1 = apply_model_on_3dimage(model_warped, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,output_path+prefix+'warped_T2_to_T1.nii.gz')
