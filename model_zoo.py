import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, BatchNormalization, Activation, Input, Add, Lambda
from keras.layers import UpSampling3D, MaxPooling3D, Subtract, GlobalAveragePooling3D, Reshape
from keras.layers import concatenate, Flatten
from keras.regularizers import l1, l2
from keras.constraints import max_norm

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


#def build_feature_model(init_shape, n_filters=32, kernel_size=3):
#  if K.image_data_format() == 'channels_first':
#    channel_axis = 1
#  else:
#    channel_axis = -1
#      
#  inputs = Input(shape=init_shape)
#  
#  features = inputs
#      
#  features = Conv3D(n_filters, (kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
#                  use_bias=False,
#                  strides=1)(features)
#  features = BatchNormalization(axis=channel_axis)(features)
#  features = Activation('tanh')(features)
#  
#  model = Model(inputs=inputs, outputs=features)
#  return model

def build_feature_model(init_shape, n_filters=32, kernel_size=3):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
      
  inputs = Input(shape=init_shape)
  
  features = inputs
      
  features = Conv3D((int)(n_filters/2), (kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
                  use_bias=False,
                  strides=1)(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('relu')(features) 
  
  features = Conv3D(filters=n_filters, kernel_size=(3, 3, 3), strides=2, use_bias=False, padding='same')(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('tanh')(features)
  
  model = Model(inputs=inputs, outputs=features)
  return model


#def build_recon_model(init_shape, n_channelsY=1, n_filters=32, kernel_size=3):
#  input_recon = Input(shape=init_shape)
#
#  recon = Activation('tanh')(input_recon)
#  
#  recon = Conv3D(n_channelsY, (kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='he_normal',
#                  use_bias=False,
#                  strides=1)(recon)
#  
#  recon = Activation('linear')(recon)
#  
#  model = Model(inputs=input_recon, outputs=recon)
#  return model

def build_recon_model(init_shape, n_channelsY=1, n_filters=32, kernel_size=3):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
  
  input_recon = Input(shape=init_shape)

  recon=(UpSampling3D((2, 2, 2)))(input_recon)  
  recon = Conv3D(filters=n_filters, kernel_size=(3, 3, 3), strides=1, use_bias=False, padding='same')(recon)
  recon = BatchNormalization(axis=channel_axis)(recon)
  recon = Activation('relu')(recon)
  
  recon = Conv3D(filters=n_channelsY, kernel_size=(kernel_size, kernel_size, kernel_size), strides=1, use_bias=False, padding='same')(recon)
#  recon=(Activation('linear'))(recon)
  
  model = Model(inputs=input_recon, outputs=recon)
  return model

def build_block_model(init_shape, n_filters=32, n_layers=2):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
  
  input_block = Input(shape=init_shape)
  x = input_block
  
  block_type = 'res_in_res'
  ratio = 1 #change inside the block the number of filters

  if block_type == 'resnet':    
    for i in range(n_layers-1):
      x = Conv3D((int)(n_filters*ratio), (3, 3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False,
                      strides=1)(x)
      x = Activation('relu')(x)
  
    x = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer='he_normal',
                    use_bias=False,
                    strides=1)(x)
    #x = Activation('tanh')(x)  
  

  if block_type == 'dense':
    for i in range(n_layers):
      #conv
      y = Conv3D((int)(n_filters*ratio), (3, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
      x = concatenate([x,y], axis=channel_axis)

      #bottleneck to keep the same feature dimension
      x = Conv3D(n_filters, (1, 1, 1), activation='relu', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    

  if block_type =='res_in_res':
    for i in range(n_layers):
      y = Conv3D((int)(n_filters*ratio), (3, 3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False,
                      strides=1,
                      kernel_constraint = max_norm(max_value=1, axis=[0,1,2]))(x)
      y = Activation('relu')(y)
  
      y = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer='he_normal',
                    use_bias=False,
                    strides=1,
                    kernel_constraint = max_norm(max_value=1, axis=[0,1,2]))(y)
      #y = Activation('tanh')(y)   #Limit the max/min fo the added residual 
      x = Add()([y,x])  
      
      
  output_block = x  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def build_reversible_forward_model(init_shape, block_f, block_g, n_layers, scaling=None):
  input_block = Input(shape=init_shape)

  if scaling is None:
    scaling = 1.0/n_layers
    print('Scaling in forward model : '+str(scaling))
    
  x = input_block

  #Split channels
  if K.image_data_format() == 'channels_last':
    x1 = Lambda(lambda x: x[:,:,:,:,:(int)(K.int_shape(x)[-1]/2)])(x)
    x2 = Lambda(lambda x: x[:,:,:,:,(int)(K.int_shape(x)[-1]/2):])(x)
    channel_axis = -1
  else:
    x1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:,:])(x)
    x2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:,:])(x)
    channel_axis = 1


  for i in range(n_layers):    
    
    xx = block_f(x2)
    y1= Add()([xx,x1])
    
    xx = block_g(y1)
    y2= Add()([xx,x2])

    x1 = y1
    x2 = y2    

  x = concatenate([x1,x2], axis=channel_axis)    
    
  #x = Activation('tanh')(x)#Limit the max/min fo the added residual
  output_block = x
  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def build_reversible_backward_model(init_shape, block_f, block_g, n_layers, scaling=None):
  input_block = Input(shape=init_shape)

  if scaling is None:
    scaling = 1.0/n_layers
    print('Scaling in backward model : '+str(scaling))
    
  x = input_block

  #Split channels
  if K.image_data_format() == 'channels_last':
    x1 = Lambda(lambda x: x[:,:,:,:,:(int)(K.int_shape(x)[-1]/2)])(x)
    x2 = Lambda(lambda x: x[:,:,:,:,(int)(K.int_shape(x)[-1]/2):])(x)
    channel_axis = -1
  else:
    x1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:,:])(x)
    x2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:,:])(x)
    channel_axis = 1


  for i in range(n_layers):    
    
    xx = block_g(x1)
    y2= Subtract()([x2,xx])
    
    xx = block_g(y2)
    y1= Subtract()([x1,xx])

    x1 = y1
    x2 = y2    

  x = concatenate([x1,x2], axis=channel_axis)    
    
  #x = Activation('tanh')(x)#Limit the max/min fo the added residual
  output_block = x
  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def build_forward_model(init_shape, block_model, n_layers, scaling=None):
  input_block = Input(shape=init_shape)

  if scaling is None:
    scaling = 1.0/n_layers
    print('Scaling in forward model : '+str(scaling))
    
  x = input_block
  for i in range(n_layers):
    xx = block_model(x)
    #Add scaling for multiresolution optimization
    xx = Lambda(lambda x: x * scaling)(xx)
    x = Add()([xx,x])     
    
    
  #x = Activation('tanh')(x)#Limit the max/min fo the added residual
  output_block = x
  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def build_backward_model(init_shape, block_model, n_layers):
  input_block = Input(shape=init_shape)

  x = input_block
  for i in range(n_layers):
    xx = block_model(x)
    x = Subtract()([x,xx])   
  output_block = x
  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def mapping_composition(init_shape, mappings):
  ix = Input(shape=init_shape)	
  ox = ix
  for i in range(len(mappings)):
    ox = mappings[i](ox)
  model = Model(inputs=ix, outputs=ox)
  return model  

def build_one_model(init_shape, feature_model, mapping_model, reconstruction_model):
  ix = Input(shape=init_shape)	
  if feature_model is not None:
    fx = feature_model(ix)   
  else:
    fx = ix
  if mapping_model is not None:
    mx = mapping_model(fx)
  else:
    mx = fx
  if reconstruction_model is not None:  
    rx = reconstruction_model(mx)
  else:
    rx = mx    
  model = Model(inputs=ix, outputs=rx)
  return model

def build_exp_model(init_shape, feature_model, mapping_model, reconstruction_model):
  ix = Input(shape=init_shape)	
  iy = Input(shape=init_shape)	
  fx = feature_model(ix)   
  fy = feature_model(iy)
     
  m = mapping_model(fx)
  r = reconstruction_model(m)
 
  rx =  reconstruction_model(fx)
  ry =  reconstruction_model(fy)
  
  model = Model(inputs=[ix,iy], outputs=[rx,ry,r])
  return model

def build_4_model(init_shape, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model):
  ix = Input(shape=init_shape)	
  iy = Input(shape=init_shape)	
  
  fx = feature_model(ix)   
  fy = feature_model(iy)
  
  #Forward     
  mx2y = mapping_x_to_y(fx)
  rx2y = reconstruction_model(mx2y)
  #Backward
  my2x = mapping_y_to_x(fy)
  ry2x = reconstruction_model(my2x)

  #Identity constraints
  my2y = mapping_x_to_y(fy)
  ry2y = reconstruction_model(my2y)
  mx2x = mapping_y_to_x(fx)
  rx2x = reconstruction_model(mx2x)
  
  model = Model(inputs=[ix,iy], outputs=[rx2y,ry2x,rx2x,ry2y])
  return model



  
def fm_model(init_shape, feature_model, mapping_model):
  ix = Input(shape=init_shape)	
  iy = Input(shape=init_shape)	
  fx = feature_model(ix)   
  fy = feature_model(iy)
     
  m = mapping_model(fx)
  
  err = Subtract()([fy,m])
  err = Lambda(lambda x:K.abs(x))(err)
  err = GlobalAveragePooling3D()(err)
  err = Lambda(lambda x: K.mean(x, axis=1))(err) 
  err = Reshape((1,))(err)

  model = Model(inputs=[ix,iy], outputs=err)
  return model  
  
def TriangleModel(init_shape, feature_shape, feature_model, block_PD_to_T2, block_T2_to_T1, block_T1_to_PD, reconstruction_model, n_layers):

  inputT1 = Input(shape=init_shape)	
  inputT2 = Input(shape=init_shape)	
  inputPD = Input(shape=init_shape)	
    
  fT1 = feature_model(inputT1)    
  fT2 = feature_model(inputT2)
  fPD = feature_model(inputPD)  
  
  forward_PD_to_T2 = build_forward_model(init_shape = feature_shape, block_model = block_PD_to_T2, n_layers = n_layers)
  forward_T2_to_T1 = build_forward_model(init_shape = feature_shape, block_model = block_T2_to_T1, n_layers = n_layers)
  forward_T1_to_PD = build_forward_model(init_shape = feature_shape, block_model = block_T1_to_PD, n_layers = n_layers)
  
  predT2 = reconstruction_model( forward_PD_to_T2 (fPD) )
  predT1 = reconstruction_model( forward_T2_to_T1 (fT2) )
  predPD = reconstruction_model( forward_T1_to_PD (fT1) )

  
  errT2 = Subtract()([inputT2,predT2])
  errT2 = Lambda(lambda x:K.abs(x))(errT2)
  errT2 = GlobalAveragePooling3D()(errT2)
  errT2 = Reshape((1,))(errT2)

  errT1 = Subtract()([inputT1,predT1])
  errT1 = Lambda(lambda x:K.abs(x))(errT1)  
  errT1 = GlobalAveragePooling3D()(errT1)
  errT1 = Reshape((1,))(errT1)

  errPD = Subtract()([inputPD,predPD])
  errPD = Lambda(lambda x:K.abs(x))(errPD)    
  errPD = GlobalAveragePooling3D()(errPD)
  errPD = Reshape((1,))(errPD)
  
  errsum = Add()([errT2, errT1, errPD])
  
  model = Model(inputs=[inputT1, inputT2, inputPD], outputs=errsum)
   
  
##  model = Model(inputs=[inputT1,inputT2,inputPD], outputs=errsum)
#  model = Model(inputs=inputT1, outputs=errsum)
  
  return model  

  