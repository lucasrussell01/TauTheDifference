import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, Activation, BatchNormalization, Flatten, \
                                    Concatenate, PReLU, MaxPooling2D
                                    
                                    
def layer_ending(layer, n, dim2d = True, dropout=0): 
    # TODO: Specify activation Function in Config
    norm_layer = BatchNormalization(name="norm_{}".format(n))(layer)
    if dim2d: # If Convolutional or Pooling Layer
        activation_layer = PReLU(shared_axes=[1, 2], name='activation_{}'.format(n))(norm_layer)
    else:
        activation_layer = PReLU(name='activation_{}'.format(n))(norm_layer)
    if dropout!=0:
        final = Dropout(dropout, name="dropout_{}".format(n))(activation_layer)
        return final
    else: 
        return activation_layer 
   
def conv_block(prev_layer, channels, kernel_size=3, n=1, dropout=0):
    # TODO: Add Kernel Regularizer
    conv = Conv2D(channels, kernel_size, name="conv_{}".format(n),
                  activation='relu', kernel_initializer='he_uniform')(prev_layer) # kernel_regularizer=None (no reg for now)
    out = layer_ending(conv, n, dropout=dropout)
    return out

def pool_block(prev_layer, n=1, dropout=0):
    poolsize = 3
    pool = MaxPooling2D(pool_size = poolsize, strides=poolsize, name="maxpooling_{}".format(n))(prev_layer)
    out = layer_ending(pool, n, dropout=dropout)
    return out

def dense_block(prev_layer, size, n=1, dropout=0):
    # TODO: Specify kernel init in config
    dense = Dense(size, name="dense_{}".format(n), kernel_initializer='he_uniform')(prev_layer)
    out = layer_ending(dense, n, dim2d=False, dropout=dropout)
    return out