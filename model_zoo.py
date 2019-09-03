import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, BatchNormalization, Activation, Input, Add
from keras.layers import UpSampling3D, MaxPooling3D
from keras.regularizers import l1, l2

def conv_bn_relu(input,n_filters=128,strides=1):
  
  weight_decay = 0.0005

  x = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer='he_normal',
                  kernel_regularizer=l2(weight_decay),
                  use_bias=False,
                  strides=strides)(input)

  channel_axis = 1 if K.image_data_format() == "channels_first" else -1
  x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
  x = Activation('relu')(x)
  return x


def ResNet3D(n_channelsX, n_channelsY, n_filters=32, n_layers=5):
  if K.image_data_format() == 'channels_first':
    init_shape = (n_channelsX, None, None, None)
    channel_axis = 1
  else:
    init_shape = (None, None, None,n_channelsX)
    channel_axis = -1

  in1 = Input(shape=init_shape, name='in1')	
  
  x = Conv3D(n_filters, (3, 3, 3), padding='same')(in1)
  x = BatchNormalization(axis=channel_axis)(x)
  x = Activation('relu')(x)

  for i in range(n_layers-1):
    y = conv_bn_relu(input=x,n_filters=n_filters)
    y = conv_bn_relu(input=y,n_filters=n_filters)
    x = Add()([x,y])

  x=(Conv3D(n_channelsY, (3, 3, 3), activation=None, padding='same'))(x)
  output=(Activation('linear'))(x)
  
  model =  Model(in1,output)
  return model   

def AutoEncoder3D(n_channelsX, n_channelsY, n_filters=32):
  if K.image_data_format() == 'channels_first':
    init_shape = (n_channelsX, None, None, None)
    channel_axis = 1
  else:
    init_shape = (None, None, None,n_channelsX)
    channel_axis = -1
  
  in1 = Input(shape=init_shape, name='in1')	

  x = Conv3D(n_filters, (3, 3, 3), padding='same')(in1)
  x = BatchNormalization(axis=channel_axis)(x)
  x = Activation('relu')(x)
  
  x = conv_bn_relu(input=x,n_filters=n_filters)
  x = conv_bn_relu(input=x,n_filters=n_filters)
    
  #MaxPooling for dimensionality reduction
  x=(MaxPooling3D((2, 2, 2), padding='valid'))(x)
  
  x = conv_bn_relu(input=x,n_filters=n_filters)
  x = conv_bn_relu(input=x,n_filters=n_filters)
  x = conv_bn_relu(input=x,n_filters=n_filters)
  	
  #Upsampling to get back to the original dimensions
  #This syntax only works with Theano, not TensorFlow !!!
  x=(UpSampling3D((2, 2, 2)))(x)
  
  x = conv_bn_relu(input=x,n_filters=n_filters)
  	
  x=(Conv3D(n_channelsY, (3, 3, 3), activation=None, padding='same'))(x)
  output=(Activation('linear'))(x)
  
  model =  Model(in1,output)
  return model 

def CHP(n_channelsX=1, n_channelsY=1):
  if K.image_data_format() == 'channels_first':
    init_shape = (n_channelsX, None, None, None)
    channel_axis = 1
  else:
    init_shape = (None, None, None,n_channelsX)
    channel_axis = -1
  
  in1 = Input(shape=init_shape, name='in1')	
  
  x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same')(in1)
  x = BatchNormalization(axis=channel_axis)(x)
  x = Activation('relu')(x)
  
  x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=2, use_bias=False, padding='same')(x)
  x = BatchNormalization(axis=channel_axis)(x)
  x = Activation('relu')(x)
  
  for i in range(6):
    y = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same')(x)
    y = BatchNormalization(axis=channel_axis)(y)
    y = Activation('relu')(y)
    y = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same')(y)
    y = BatchNormalization(axis=channel_axis)(y)
    x = Add()([x,y])
  
  x=(UpSampling3D((2, 2, 2)))(x)  
  x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same')(x)
  x = BatchNormalization(axis=channel_axis)(x)
  x = Activation('relu')(x)
  
  x = Conv3D(filters=n_channelsY, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same')(x)
  output=(Activation('linear'))(x)
  
  model =  Model(in1,output)
  return model   
  