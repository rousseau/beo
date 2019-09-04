from dataset import get_ixi_3dpatches
from model_zoo import ResNet3D, AutoEncoder3D, CHP, TriangleModel
from model_zoo import build_feature_model, build_recon_model, build_block_model
from model_zoo import build_forward_model, build_backward_model, build_one_model, mapping_composition
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import nibabel
from utils import apply_model_on_3dimage

print(K.image_data_format())

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
n_layers = 10
learning_rate = 0.0001
loss = 'mae'
batch_size = 16
epochs = 10

if K.image_data_format() == 'channels_first':
  init_shape = (n_channelsX, None, None, None)
  feature_shape = (n_filters, None, None, None)
  channel_axis = 1
else:
  init_shape = (None, None, None,n_channelsX)
  feature_shape = (None, None, None, n_filters)
  channel_axis = -1
  
feature_model = build_feature_model(init_shape=init_shape, n_filters=n_filters, kernel_size=3)
recon_model = build_recon_model(init_shape=feature_shape, n_channelsY=n_channelsY, n_filters=n_filters, kernel_size=3)

block_PD_to_T2 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=2) 
block_T2_to_T1 =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=2) 
block_T1_to_PD =  build_block_model(init_shape=feature_shape, n_filters=n_filters, n_layers=2) 
  
#triad_model = TriangleModel(init_shape = init_shape, 
#                            feature_shape = feature_shape,
#                            feature_model = feature_model,
#                            block_PD_to_T2 = block_PD_to_T2,
#                            block_T2_to_T1 = block_T2_to_T1,
#                            block_T1_to_PD = block_T1_to_PD,
#                            reconstruction_model = recon_model,
#                            n_layers = n_layers)
#triad_model.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
#triad_model.summary()
#hist = triad_model.fit(x=[T1,T2,PD], y=np.zeros((T1.shape[0])), batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

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

#Identity consistency
model_identity = build_one_model(init_shape, feature_model = feature_model, mapping_model = None, reconstruction_model = recon_model)

model_identity.compile(optimizer=Adam(lr=learning_rate), loss=loss) 

inverse_consistency = 0
cycle_consistency = 0
identity_consistency = 0

#for e in range(epochs):
#  print('Direct mappings')
#  model_PD_to_T2.fit(x=PD, y=T2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#  model_T2_to_T1.fit(x=T2, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#  model_T1_to_PD.fit(x=T1, y=PD, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#
#  if inverse_consistency == 1:
#    print('Inverse mappings')
#    model_T2_to_PD.fit(x=T2, y=PD, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    model_T1_to_T2.fit(x=T1, y=T2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    model_PD_to_T1.fit(x=PD, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    
#  if cycle_consistency == 1: #Cycle consistency requires three times more GPU RAM than direct mapping
#    print('Cycle mappings')
#    model_cycle_PD.fit(x=PD, y=PD, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    model_cycle_T2.fit(x=T2, y=T2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    model_cycle_T1.fit(x=T1, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#
#  if identity_consistency == 1:
#    print('Identity mappings')
#    model_identity.fit(x=PD, y=PD, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    model_identity.fit(x=T2, y=T2, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    
#    model_identity.fit(x=T1, y=T1, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)    



T1image = nibabel.load('/home/rousseau/Sync/Experiments/IXI/IXI661-HH-2788-T1.nii.gz')
T2image = nibabel.load('/home/rousseau/Sync/Experiments/IXI/IXI661-HH-2788-T2.nii.gz')
PDimage = nibabel.load('/home/rousseau/Sync/Experiments/IXI/IXI661-HH-2788-PD.nii.gz')
maskarray = nibabel.load('/home/rousseau/Sync/Experiments/IXI/IXI661-HH-2788-T1_bet_mask.nii.gz').get_data().astype(float)

prefix = 'iclr_nowindowing'
if inverse_consistency == 1:
  prefix += '_invc'
if cycle_consistency == 1:
  prefix += '_cycle'
if identity_consistency == 1:
  prefix += '_idc'  
prefix+= '_e'+str(epochs)+'_ps'+str(patch_size)+'_mp'+str(n_patches)+'_np'+str(T1.shape[0])
prefix+= '_lr'+str(learning_rate)
prefix+= '_nl'+str(n_layers)
prefix+= '_'

CHPmodel_T2_to_T1 = CHP()
CHPmodel_T2_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
CHPmodel_T2_to_T1.summary()
CHPmodel_T2_to_T1.fit(x=T2, y=T1, batch_size=batch_size, epochs=5, verbose=1, shuffle=True)
image_T2_to_T1 = apply_model_on_3dimage(CHPmodel_T2_to_T1, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,'/home/rousseau/Sync/Experiments/IXI/CHP_T2_to_T1.nii.gz')

#model_T1_to_T2 = AutoEncoder3D(n_channelsX, n_channelsY, n_filters=n_filters)
#model_T1_to_T2 = ResNet3D(n_channelsX, n_channelsY, n_filters=n_filters, n_layers=n_layers)
#model_T1_to_T2 = CHP()
#model_T1_to_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
#model_T1_to_T2.summary()
#hist = model_T1_to_T2.fit(x=T1, y=T2, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
#model_T1_to_T2.save('/home/rousseau/Sync/Experiments/IXI/'+prefix+'t1_to_t2.h5')


#model_T2_to_PD = CHP()
#model_T2_to_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
#hist = model_T2_to_PD.fit(x=T2, y=PD, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
#model_T2_to_PD.save('/home/rousseau/Sync/Experiments/IXI/'+prefix+'t2_to_pd.h5')

#model_T1_to_PD = CHP()
#model_T1_to_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
#hist = model_T1_to_PD.fit(x=T1, y=PD, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
#model_T1_to_PD.save('/home/rousseau/Sync/Experiments/IXI/'+prefix+'t1_to_pd.h5')


#Direct mapping
image_PD_to_T2 = apply_model_on_3dimage(model_PD_to_T2, PDimage, mask=maskarray)
nibabel.save(image_PD_to_T2,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'direct_PD_to_T2.nii.gz')

image_T2_to_T1 = apply_model_on_3dimage(model_T2_to_T1, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'direct_T2_to_T1.nii.gz')

image_T1_to_PD = apply_model_on_3dimage(model_T1_to_PD, T1image, mask=maskarray)
nibabel.save(image_T1_to_PD,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'direct_T1_to_PD.nii.gz')

#Inverse
image_T1_to_T2 = apply_model_on_3dimage(model_T1_to_T2,T1image,mask=maskarray)
nibabel.save(image_T1_to_T2,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'inverse_T1_to_T2.nii.gz')

image_T2_to_PD = apply_model_on_3dimage(model_T2_to_PD, T2image, mask=maskarray)
nibabel.save(image_T2_to_PD,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'inverse_T2_to_PD.nii.gz')

#Composition
image_PD_to_T2_to_T1 = apply_model_on_3dimage(model_T2_to_T1, image_PD_to_T2, mask=maskarray)
nibabel.save(image_PD_to_T2_to_T1,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'composition_PD_to_T2_to_T1.nii.gz')
 
#Cycle
image_PD_to_PD = apply_model_on_3dimage(model_cycle_PD, PDimage, mask=maskarray)
nibabel.save(image_PD_to_PD,'/home/rousseau/Sync/Experiments/IXI/'+prefix+'cycle_PD_to_PD.nii.gz')

CHPmodel_T2_to_T1 = CHP()
CHPmodel_T2_to_T1.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
CHPmodel_T2_to_T1.summary()
CHPmodel_T2_to_T1.fit(x=T2, y=T1, batch_size=batch_size, epochs=5, verbose=1, shuffle=True)
image_T2_to_T1 = apply_model_on_3dimage(CHPmodel_T2_to_T1, T2image, mask=maskarray)
nibabel.save(image_T2_to_T1,'/home/rousseau/Sync/Experiments/IXI/CHP_T2_to_T1.nii.gz')


