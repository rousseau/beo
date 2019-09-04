from dataset import get_ixi_3dpatches
from model_zoo import ResNet3D, AutoEncoder3D, CHP, TriangleModel
from model_zoo import build_feature_model, build_recon_model, build_block_model
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import nibabel
from utils import apply_model_on_3dimage
from keras.models import load_model

#Package from OpenAI for GPU memory saving
#https://github.com/cybertronai/gradient-checkpointing
import memory_saving_gradients

print(K.image_data_format())
from os.path import expanduser
home = expanduser("~")
output_path = home+'/Sync/Experiments/IXI/'

patch_size = 40
n_patches = 5000
(T1,T2,PD) = get_ixi_3dpatches(patch_size = patch_size, n_patches = n_patches)

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
n_layers = 5
learning_rate = 0.0001
loss = 'mae'
batch_size = 64
epochs = 10
use_optim = 1
inverse_consistency = 0
cycle_consistency = 0
identity_consistency = 0
load_pretrained_models = 0

if use_optim == 0:
  print('Make use of memory_saving_gradients package')
  K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

if K.image_data_format() == 'channels_first':
  init_shape = (n_channelsX, None, None, None)
  feature_shape = (n_filters, None, None, None)
  channel_axis = 1
else:
  init_shape = (None, None, None,n_channelsX)
  feature_shape = (None, None, None, n_filters)
  channel_axis = -1

#The five models required to build all the models :   
if load_pretrained_models == 0:  
  feature_model = build_feature_model(init_shape=init_shape, n_filters=n_filters, kernel_size=5)
  recon_model = build_recon_model(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=3)
else:
  feature_model = load_model(output_path+'feature_model.h5')
  recon_model = load_model(output_path+'recon_model.h5')


block_PD_to_T2 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=2) 
block_T2_to_T1 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=2) 
block_T1_to_PD =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=2) 
  

#Identity consistency
model_identity = build_one_model(init_shape, feature_model = feature_model, mapping_model = None, reconstruction_model = recon_model)
model_identity.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
feature_model.summary()
recon_model.summary()

if load_pretrained_models == 0:  
  print('Start by learning the auto-encoding part')
  print('Identity mappings')
  #The best option should be to concatenate all patches to learn over the entire set of patches
  #However, numpy concatenation requires twice the RAM
  for e in range(epochs):
    model_identity.fit(x=PD, y=PD, batch_size=int(batch_size), epochs=1, verbose=1, shuffle=True)    
    model_identity.fit(x=T2, y=T2, batch_size=int(batch_size), epochs=1, verbose=1, shuffle=True)    
    model_identity.fit(x=T1, y=T1, batch_size=int(batch_size), epochs=1, verbose=1, shuffle=True)    
  
  feature_model.save(output_path+'feature_model.h5')
  recon_model.save(output_path+'recon_model.h5')

print('Freezing the AE part of the network')
for layer in feature_model.layers:
  layer.trainable = False
feature_model.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

for layer in recon_model.layers:
  layer.trainable = False
recon_model.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
  
#check if freeze is ok
for layer in feature_model.layers:
  print(layer)
  print(layer.trainable)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

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

model_PD_to_T2.summary()

#Inverse consistency
mapping_T2_to_PD = build_backward_model(init_shape=feature_shape, block_model=block_PD_to_T2, n_layers=n_layers)
mapping_T1_to_T2 = build_backward_model(init_shape=feature_shape, block_model=block_T2_to_T1, n_layers=n_layers)
mapping_PD_to_T1 = build_backward_model(init_shape=feature_shape, block_model=block_T1_to_PD, n_layers=n_layers)

model_T2_to_PD = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_T2_to_PD, reconstruction_model = recon_model)
model_T1_to_T2 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_T1_to_T2, reconstruction_model = recon_model)
model_PD_to_T1 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_PD_to_T1, reconstruction_model = recon_model)

model_T2_to_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_T1_to_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_PD_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

#Cycle forward consistency (by mapping composition)
mapping_cycle_PD = mapping_composition(init_shape = feature_shape, mappings=[mapping_PD_to_T2,mapping_T2_to_T1,mapping_T1_to_PD])
mapping_cycle_T1 = mapping_composition(init_shape = feature_shape, mappings=[mapping_T1_to_PD,mapping_PD_to_T2,mapping_T2_to_T1])
mapping_cycle_T2 = mapping_composition(init_shape = feature_shape, mappings=[mapping_T2_to_T1,mapping_T1_to_PD,mapping_PD_to_T2])

model_cycle_PD = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_cycle_PD, reconstruction_model = recon_model)
model_cycle_T2 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_cycle_T2, reconstruction_model = recon_model)
model_cycle_T1 = build_one_model(init_shape, feature_model = feature_model, mapping_model = mapping_cycle_T1, reconstruction_model = recon_model)

model_cycle_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_cycle_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_cycle_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

#Apply on a test image
T1image = nibabel.load(output_path+'IXI661-HH-2788-T1.nii.gz')
T2image = nibabel.load(output_path+'IXI661-HH-2788-T2.nii.gz')
PDimage = nibabel.load(output_path+'IXI661-HH-2788-PD.nii.gz')
maskarray = nibabel.load(output_path+'IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)

prefix = 'iclr_nowindowing'
if use_optim == 1:
  prefix += '_optim'
if inverse_consistency == 1:
  prefix += '_invc'
if cycle_consistency == 1:
  prefix += '_cycle'
if identity_consistency == 1:
  prefix += '_idc'  
prefix+= '_e'+str(epochs)+'_ps'+str(patch_size)+'_mp'+str(n_patches)+'_np'+str(T1.shape[0])
prefix+= '_bs'+str(batch_size)
prefix+= '_lr'+str(learning_rate)
prefix+= '_nl'+str(n_layers)
prefix+= '_'

#Encoding
image_T1_id = apply_model_on_3dimage(model_identity, T1image, mask=maskarray)
nibabel.save(image_T1_id,output_path+prefix+'id_T1.nii.gz')

image_T2_id = apply_model_on_3dimage(model_identity, T2image, mask=maskarray)
nibabel.save(image_T2_id,output_path+prefix+'id_T2.nii.gz')

image_PD_id = apply_model_on_3dimage(model_identity, PDimage, mask=maskarray)
nibabel.save(image_PD_id,output_path+prefix+'id_PD.nii.gz')


for e in range(epochs):
  print('Direct mappings')
  model_PD_to_T2.fit(x=PD, y=T2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
  model_T2_to_T1.fit(x=T2, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
  model_T1_to_PD.fit(x=T1, y=PD, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    

  if inverse_consistency == 1:
    print('Inverse mappings')
    model_T2_to_PD.fit(x=T2, y=PD, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
    model_T1_to_T2.fit(x=T1, y=T2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
    model_PD_to_T1.fit(x=PD, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
    
  if cycle_consistency == 1: #Cycle consistency requires three times more GPU RAM than direct mapping
    print('Cycle mappings')
    model_cycle_PD.fit(x=PD, y=PD, batch_size=int(batch_size/3), epochs=1, verbose=1, shuffle=True)    
    model_cycle_T2.fit(x=T2, y=T2, batch_size=int(batch_size/3), epochs=1, verbose=1, shuffle=True)    
    model_cycle_T1.fit(x=T1, y=T1, batch_size=int(batch_size/3), epochs=1, verbose=1, shuffle=True)    

  if identity_consistency == 1: #no mapping so batch size could be increased
    print('Identity mappings')
    model_identity.fit(x=PD, y=PD, batch_size=int(batch_size), epochs=1, verbose=1, shuffle=True)    
    model_identity.fit(x=T2, y=T2, batch_size=int(batch_size), epochs=1, verbose=1, shuffle=True)    
    model_identity.fit(x=T1, y=T1, batch_size=int(batch_size), epochs=1, verbose=1, shuffle=True)    

#Direct mapping
image_PD_to_T2 = apply_model_on_3dimage(model_PD_to_T2, PDimage, mask=maskarray)
nibabel.save(image_PD_to_T2,output_path+prefix+'direct_PD_to_T2.nii.gz')

image_T2_to_T1 = apply_model_on_3dimage(model_T2_to_T1, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,output_path+prefix+'direct_T2_to_T1.nii.gz')

image_T1_to_PD = apply_model_on_3dimage(model_T1_to_PD, T1image, mask=maskarray)
nibabel.save(image_T1_to_PD,output_path+prefix+'direct_T1_to_PD.nii.gz')

#Inverse
image_T1_to_T2 = apply_model_on_3dimage(model_T1_to_T2,T1image,mask=maskarray)
nibabel.save(image_T1_to_T2,output_path+prefix+'inverse_T1_to_T2.nii.gz')

image_T2_to_PD = apply_model_on_3dimage(model_T2_to_PD, T2image, mask=maskarray)
nibabel.save(image_T2_to_PD,output_path+prefix+'inverse_T2_to_PD.nii.gz')

#Composition
image_PD_to_T2_to_T1 = apply_model_on_3dimage(model_T2_to_T1, image_PD_to_T2, mask=maskarray)
nibabel.save(image_PD_to_T2_to_T1,output_path+prefix+'composition_PD_to_T2_to_T1.nii.gz')
 
#Cycle
image_PD_to_PD = apply_model_on_3dimage(model_cycle_PD, PDimage, mask=maskarray)
nibabel.save(image_PD_to_PD,output_path+prefix+'cycle_PD_to_PD.nii.gz')

#CHPmodel_T2_to_T1 = CHP()
#CHPmodel_T2_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
#CHPmodel_T2_to_T1.summary()
#CHPmodel_T2_to_T1.fit(x=T2, y=T1, batch_size=batch_size, epochs=5, verbose=1, shuffle=True)
#image_T2_to_T1 = apply_model_on_3dimage(CHPmodel_T2_to_T1, T2image, mask=maskarray)
#nibabel.save(image_T2_to_T1,'/home/rousseau/Sync/Experiments/IXI/CHP_T2_to_T1.nii.gz')


