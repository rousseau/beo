from dataset import get_ixi_data
from model_zoo import ResNet3D, AutoEncoder3D
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
import nibabel
from utils import array_normalization

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
epochs = 10

model_T1_to_T2 = AutoEncoder3D(n_channelsX, n_channelsY, n_filters=n_filters)
#model_T1_to_T2 = ResNet3D(n_channelsX, n_channelsY, n_filters=n_filters, n_layers=n_layers)
model_T1_to_T2.compile(optimizer=Adam(lr=learning_rate), loss=loss) 
model_T1_to_T2.summary()

hist = model_T1_to_T2.fit(x=T1, y=T2, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)

model_T1_to_T2.save('/home/rousseau/Exp/toto.h5')

T1image = nibabel.load('/home/rousseau/Exp/IXI/IXI662-Guys-1120-T1.nii.gz')
T1array = T1image.get_data().astype(float)
T1array = np.expand_dims(T1array,axis=0)

maskarray = nibabel.load('/home/rousseau/Exp/IXI/IXI662-Guys-1120-T1_bet_mask.nii.gz').get_data().astype(float)
maskarray = np.expand_dims(maskarray,axis=0)
T1array = array_normalization(X=T1array,M=maskarray,norm=0)
T1array = np.expand_dims(T1array,axis=-1)

simu = model_T1_to_T2.predict(T1array, batch_size=1)

output_image = nibabel.Nifti1Image(np.squeeze(simu[0,:,:,:,0]), T1image.affine)
nibabel.save(output_image,'/home/rousseau/Exp/toto.nii.gz')
