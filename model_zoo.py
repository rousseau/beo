import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, BatchNormalization, Activation, Input, Add, Lambda
from keras.layers import Conv2D, UpSampling2D
from keras.layers import UpSampling3D, MaxPooling3D, Subtract, GlobalAveragePooling3D, Reshape
from keras.layers import concatenate, Flatten
from keras.regularizers import l1, l2
from keras.constraints import max_norm

def conv_bn_relu(input,n_filters=128,strides=1):
  
  weight_decay = 0.0005

  x = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
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
#  features = Conv3D(n_filters, (kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='glorot_uniform',
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
      
  features = Conv3D((int)(n_filters/2), (kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('relu')(features) 
  
  features = Conv3D(filters=n_filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=2)(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('tanh')(features)
  
  model = Model(inputs=inputs, outputs=features)
  return model

def build_feature_model_2d(init_shape, n_filters=32, kernel_size=3):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
      
  inputs = Input(shape=init_shape)
  
  features = inputs
      
  features = Conv2D((int)(n_filters/2), (kernel_size, kernel_size), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('relu')(features) 
  
  features = Conv2D(filters=n_filters, kernel_size=(3,3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=2)(features)
  features = BatchNormalization(axis=channel_axis)(features)
  features = Activation('tanh')(features)
  
  model = Model(inputs=inputs, outputs=features)
  return model

#def build_recon_model(init_shape, n_channelsY=1, n_filters=32, kernel_size=3):
#  input_recon = Input(shape=init_shape)
#
#  recon = Activation('tanh')(input_recon)
#  
#  recon = Conv3D(n_channelsY, (kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='glorot_uniform',
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
  recon = Conv3D(filters=n_filters, kernel_size=(3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(recon)
  recon = BatchNormalization(axis=channel_axis)(recon)
  recon = Activation('relu')(recon)
  
  recon = Conv3D(filters=n_channelsY, kernel_size=(kernel_size, kernel_size, kernel_size), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(recon)
#  recon=(Activation('linear'))(recon)
  
  model = Model(inputs=input_recon, outputs=recon)
  return model

def build_recon_model_2d(init_shape, n_channelsY=1, n_filters=32, kernel_size=3):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
  
  input_recon = Input(shape=init_shape)

  recon=(UpSampling2D((2, 2)))(input_recon)  
  recon = Conv2D(filters=n_filters, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(recon)
  recon = BatchNormalization(axis=channel_axis)(recon)
  recon = Activation('relu')(recon)
  
  recon = Conv2D(filters=n_channelsY, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(recon)
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
      x = Conv3D((int)(n_filters*ratio), (3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
                      use_bias=False,
                      strides=1)(x)
      x = Activation('relu')(x)
  
    x = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(x)
    #x = Activation('tanh')(x)  
  

  if block_type == 'dense':
    for i in range(n_layers):
      #conv
      y = Conv3D((int)(n_filters*ratio), (3, 3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', use_bias=False)(x)
      x = concatenate([x,y], axis=channel_axis)

      #bottleneck to keep the same feature dimension
      x = Conv3D(n_filters, (1, 1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform', use_bias=False)(x)
    

  if block_type =='res_in_res':
    for i in range(n_layers):
      y = Conv3D((int)(n_filters*ratio), (3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
                      use_bias=False,
                      strides=1)(x)#,
                      #kernel_constraint = max_norm(max_value=1, axis=[0,1,2]))(x)
      y = Activation('relu')(y)
  
      y = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(y)#,
                    #kernel_constraint = max_norm(max_value=1, axis=[0,1,2]))(y)
      #y = Activation('tanh')(y)   #Limit the max/min fo the added residual 
      x = Add()([y,x])  
      
      
  output_block = x  
  model = Model(inputs=input_block, outputs=output_block)
  return model


def build_block_model_2d(init_shape, n_filters=32, n_layers=2):
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
      x = Conv2D((int)(n_filters*ratio), (3, 3), padding='same', kernel_initializer='glorot_uniform',
                      use_bias=False,
                      strides=1)(x)
      x = Activation('relu')(x)
  
    x = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(x)
    #x = Activation('tanh')(x)  
  

  if block_type == 'dense':
    for i in range(n_layers):
      #conv
      y = Conv2D((int)(n_filters*ratio), (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', use_bias=False)(x)
      x = concatenate([x,y], axis=channel_axis)

      #bottleneck to keep the same feature dimension
      x = Conv2D(n_filters, (1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform', use_bias=False)(x)
    

  if block_type =='res_in_res':
    for i in range(n_layers):
      y = Conv2D((int)(n_filters*ratio), (3, 3), padding='same', kernel_initializer='glorot_uniform',
                      use_bias=False,
                      strides=1)(x)#,
                      #kernel_constraint = max_norm(max_value=1, axis=[0,1,2]))(x)
      y = Activation('relu')(y)
  
      y = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer='glorot_uniform',
                    use_bias=False,
                    strides=1)(y)#,
                    #kernel_constraint = max_norm(max_value=1, axis=[0,1,2]))(y)
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
    y1= Add()([x1,xx])
    
    xx = block_g(y1)
    y2= Add()([x2,xx])

    x1 = y1
    x2 = y2    

  x = concatenate([x1,x2], axis=channel_axis)    
    
  #x = Activation('tanh')(x)#Limit the max/min fo the added residual
  output_block = x
  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def build_reversible_forward_model_2d(init_shape, block_f, block_g, n_layers, scaling=None):
  input_block = Input(shape=init_shape)

  if scaling is None:
    scaling = 1.0/n_layers
    print('Scaling in forward model : '+str(scaling))
    
  x = input_block

  #Split channels
  if K.image_data_format() == 'channels_last':
    x1 = Lambda(lambda x: x[:,:,:,:(int)(K.int_shape(x)[-1]/2)])(x)
    x2 = Lambda(lambda x: x[:,:,:,(int)(K.int_shape(x)[-1]/2):])(x)
    channel_axis = -1
  else:
    x1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:])(x)
    x2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:])(x)
    channel_axis = 1


  for i in range(n_layers):    
    
    xx = block_f[i](x2)
    y1= Add()([x1,xx])
    
    xx = block_g[i](y1)
    y2= Add()([x2,xx])

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
    
  y = input_block

  #Split channels
  if K.image_data_format() == 'channels_last':
    y1 = Lambda(lambda x: x[:,:,:,:,:(int)(K.int_shape(x)[-1]/2)])(y)
    y2 = Lambda(lambda x: x[:,:,:,:,(int)(K.int_shape(x)[-1]/2):])(y)
    channel_axis = -1
  else:
    y1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:,:])(y)
    y2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:,:])(y)
    channel_axis = 1


  for i in range(n_layers):    
    
    yy = block_g(y1)
    x2= Subtract()([y2,yy])
    
    yy = block_f(x2)
    x1= Subtract()([y1,yy])

    y1 = x1
    y2 = x2    

  y = concatenate([y1,y2], axis=channel_axis)    
    
  #x = Activation('tanh')(x)#Limit the max/min fo the added residual
  output_block = y
  
  model = Model(inputs=input_block, outputs=output_block)
  return model

def build_reversible_backward_model_2d(init_shape, block_f, block_g, n_layers, scaling=None):
  input_block = Input(shape=init_shape)

  if scaling is None:
    scaling = 1.0/n_layers
    print('Scaling in backward model : '+str(scaling))
    
  y = input_block

  #Split channels
  if K.image_data_format() == 'channels_last':
    y1 = Lambda(lambda x: x[:,:,:,:(int)(K.int_shape(x)[-1]/2)])(y)
    y2 = Lambda(lambda x: x[:,:,:,(int)(K.int_shape(x)[-1]/2):])(y)
    channel_axis = -1
  else:
    y1 = Lambda(lambda x: x[:,:(int)(K.int_shape(x)[-1]/2),:,:])(y)
    y2 = Lambda(lambda x: x[:,(int)(K.int_shape(x)[-1]/2):,:,:])(y)
    channel_axis = 1


  for i in range(n_layers):    
    
    yy = block_g[n_layers-1-i](y1)
    x2= Subtract()([y2,yy])
    
    yy = block_f[n_layers-1-i](x2)
    x1= Subtract()([y1,yy])

    y1 = x1
    y2 = x2    

  y = concatenate([y1,y2], axis=channel_axis)    
    
  #x = Activation('tanh')(x)#Limit the max/min fo the added residual
  output_block = y
  
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


def build_model(init_shape, feature_model, mapping_x_to_y, mapping_y_to_x, reconstruction_model, mode=0):
  #mode:
  #0 : direct mapping only
  #1 : inverse mapping only
  #2 : direct and inverse mapping
  #3 : direct and identity (through mapping) 
  #4 : direct and identity for x
  #5 : direct and cycle consistency for x
  #6 : 5 + identity for x
  #7 : direct, inverse, cycle
  #8 : autoencoder for x
  #9 : autoencoder + direct mapping
  
  ix = Input(shape=init_shape)
  iy = Input(shape=init_shape)

  fx = feature_model(ix) # features of x
  fy = feature_model(iy) # features of y

  mx2y = mapping_x_to_y(fx) # direct mapping
  my2x = mapping_y_to_x(fy) # inverse mapping

  rx2y = reconstruction_model(mx2y) # reconstruction of direct mapping
  ry2x = reconstruction_model(my2x) # reconstruction of inverse mapping

  #identity through mapping
  mx2x = mapping_y_to_x(fx)
  rx2x = reconstruction_model(mx2x)

  my2y = mapping_x_to_y(fy)
  ry2y = reconstruction_model(my2y)

  #autoencoder
  rx = reconstruction_model(fx)
  ry = reconstruction_model(fy)

  #cycle consistency
  fx2y = feature_model(rx2y)
  mx2y2x = mapping_y_to_x(fx2y)
  rx2y2x = reconstruction_model(mx2y2x) 

  fy2x = feature_model(ry2x)
  my2x2y = mapping_x_to_y(fy2x)
  ry2x2y = reconstruction_model(my2x2y)  

  if mode == -1:  
    return Model(inputs=ix, outputs=fx)

  if mode == 0:
    return Model(inputs=ix, outputs=rx2y)

  if mode == 1:
    return Model(inputs=iy, outputs=ry2x)

  if mode == 2:
    return Model(inputs=[ix,iy], outputs=[rx2y,ry2x])

  if mode == 3:
    return Model(inputs=[ix,iy], outputs=[rx2y,ry2y])

  if mode == 4:
    return Model(inputs=ix, outputs=[rx2y,rx])

  if mode == 5:
    return Model(inputs=ix, outputs=[rx2y,rx2y2x])

  if mode == 6:
    return Model(inputs=ix, outputs=[rx2y,rx2y2x,rx])

  if mode == 7:  
    return Model(inputs=[ix,iy], outputs=[rx2y,ry2x,rx2y2x,ry2x2y])  

  if mode == 8:
    return Model(inputs=ix, outputs=rx)  

  if mode == 9:
    return Model(inputs=ix, outputs=[rx2y,rx])


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

  #Identity mapping constraints
  my2y = mapping_x_to_y(fy)
  ry2y = reconstruction_model(my2y)

  mx2x = mapping_y_to_x(fx)
  rx2x = reconstruction_model(mx2x) 

  #Cycle consistency
  frx2y = feature_model(rx2y)
  mx2y2x = mapping_y_to_x(frx2y)
  rx2y2x = reconstruction_model(mx2y2x)  

  fry2x = feature_model(ry2x)
  my2x2y = mapping_x_to_y(fry2x)
  ry2x2y = reconstruction_model(my2x2y)  

  #Autoencoder constraints
  idy2y = reconstruction_model(fy)
  idx2x = reconstruction_model(fx)  

  #Errors  
  # errx2y = Subtract()([rx2y,iy])
  # errx2y = Lambda(lambda x:K.pow(x,2))(errx2y)
  # errx2y = GlobalAveragePooling3D()(errx2y)
  # errx2y = Reshape((1,))(errx2y)

  # erry2x = Subtract()([ry2x,ix])
  # erry2x = Lambda(lambda x:K.pow(x,2))(erry2x)
  # erry2x = GlobalAveragePooling3D()(erry2x)
  # erry2x = Reshape((1,))(erry2x)
  
  # errx2x = Subtract()([rx2x,ix])
  # errx2x = Lambda(lambda x:K.pow(x,2))(errx2x)
  # errx2x = GlobalAveragePooling3D()(errx2x)
  # errx2x = Reshape((1,))(errx2x)
  
  # erry2y = Subtract()([ry2y,iy])
  # erry2y = Lambda(lambda x:K.pow(x,2))(erry2y)
  # erry2y = GlobalAveragePooling3D()(erry2y)
  # erry2y = Reshape((1,))(erry2y)
  
  # erridx = Subtract()([idx2x,ix])
  # erridx = Lambda(lambda x:K.abs(x))(erridx)
  # erridx = GlobalAveragePooling3D()(erridx)
  # erridx = Reshape((1,))(erridx)

  # erridy = Subtract()([idy2y,iy])
  # erridy = Lambda(lambda x:K.abs(x))(erridy)
  # erridy = GlobalAveragePooling3D()(erridy)
  # erridy = Reshape((1,))(erridy)  
  
  # errmfx = Subtract()([fy,mx2y])
  # errmfx = Lambda(lambda x:K.abs(x))(errmfx)
  # errmfx = GlobalAveragePooling3D()(errmfx)
  # errmfx = Lambda(lambda x: K.mean(x, axis=1))(errmfx) 
  # errmfx = Reshape((1,))(errmfx)
  
  # errmfy = Subtract()([fx,my2x])
  # errmfy = Lambda(lambda x:K.abs(x))(errmfy)
  # errmfy = GlobalAveragePooling3D()(errmfy)
  # errmfy = Lambda(lambda x: K.mean(x, axis=1))(errmfy)  
  # errmfy = Reshape((1,))(errmfy)
  
  
#  errsum = Add()([errx2y, erry2x, errx2x, erry2y, erridx, erridy, errmfx, errmfy])
#  errsum = Add()([errx2y, erry2x, errx2x, erry2y])

#  model = Model(inputs=[ix,iy], outputs=[rx2y,ry2x,rx2x,ry2y,idx2x,idy2y])
  model = Model(inputs=[ix,iy], outputs=[rx2y,ry2x,rx2x,ry2y,rx2y2x,ry2x2y,idx2x,idy2y])
#  model = Model(inputs=[ix,iy], outputs=errsum)
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



  