from dataset import get_ixi_data
from model_zoo import ResNet3D, AutoEncoder3D, CHP
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import nibabel
from utils import apply_model_on_3dimage

print(K.image_data_format())

patch_size = 40
n_patches = 1000
(T1,T2,PD) = get_ixi_data(patch_size = patch_size, n_patches = n_patches)

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
learning_rate = 0.001
loss = 'mae'
batch_size = 32
epochs = 5


T1image = nibabel.load('/home/rousseau/Exp/IXI/IXI662-Guys-1120-T1.nii.gz')
T2image = nibabel.load('/home/rousseau/Exp/IXI/IXI662-Guys-1120-T2.nii.gz')
PDimage = nibabel.load('/home/rousseau/Exp/IXI/IXI662-Guys-1120-PD.nii.gz')
maskarray = nibabel.load('/home/rousseau/Exp/IXI/IXI662-Guys-1120-T1_bet_mask.nii.gz').get_data().astype(float)

#model_T1_to_T2 = AutoEncoder3D(n_channelsX, n_channelsY, n_filters=n_filters)
#model_T1_to_T2 = ResNet3D(n_channelsX, n_channelsY, n_filters=n_filters, n_layers=n_layers)
model_T1_to_T2 = CHP()
model_T1_to_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_T1_to_T2.summary()
hist = model_T1_to_T2.fit(x=T1, y=T2, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
model_T1_to_T2.save('/home/rousseau/Exp/t1_to_t2.h5')

image_T1_to_T2 = apply_model_on_3dimage(model_T1_to_T2,T1image,mask=maskarray)
nibabel.save(image_T1_to_T2,'/home/rousseau/Exp/T1_to_T2.nii.gz')

model_T2_to_PD = CHP()
model_T2_to_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
hist = model_T2_to_PD.fit(x=T2, y=PD, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
model_T2_to_PD.save('/home/rousseau/Exp/t2_to_pd.h5')
image_T2_to_PD = apply_model_on_3dimage(model_T2_to_PD, T2image, mask=maskarray)
nibabel.save(image_T2_to_PD,'/home/rousseau/Exp/T2_to_PD.nii.gz')

model_T1_to_PD = CHP()
model_T1_to_PD.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
hist = model_T1_to_PD.fit(x=T1, y=PD, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
model_T1_to_PD.save('/home/rousseau/Exp/t1_to_pd.h5')
image_T1_to_PD = apply_model_on_3dimage(model_T1_to_PD, T1image, mask=maskarray)
nibabel.save(image_T1_to_PD,'/home/rousseau/Exp/T1_to_PD.nii.gz')

image_T1_to_T2_to_PD = apply_model_on_3dimage(model_T2_to_PD, image_T1_to_T2, mask=maskarray)
nibabel.save(image_T1_to_T2_to_PD,'/home/rousseau/Exp/T1_to_T2_to_PD.nii.gz')
