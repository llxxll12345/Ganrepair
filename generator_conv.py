from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.layers import merge
from keras.layers import Add
from keras.layers import DepthwiseConv2D
from keras.layers import UpSampling2D
from keras import backend as K

# An attemp to build a net work for generator with input reinforcement
# input the contour strengenthend image and color strengenthend image

"""
Architecture:

input (320 * 320 * 3) [contour]        => (160 * 160 * 16) => (160 * 160 * 32)                                    

input (320 * 320 * 3) [image]          => (160 * 160 * 16) => (160 * 160 * 32) => Concatenate => (160 * 160 * 32) -> (80 * 80 * 128) -> UNSAMPLE LAYERS

input (320 * 320 * 3) [color]          => (160 * 160 * 16) => (160 * 160 * 32)

"""

def getName(name, branch=None, prefix=None):
    if prefix is None:
        return None
    return prefix + '_' + name


def bottleneckBlock(self, inputs, filters, exp_factor, activation='leakyRelu', strides=1, layerName = ''):
    expanded = K.int_shape(inputs)[3] * exp_factor
    layer = conv2dbn(inputs, expanded, 1, strides=1, layerName= getName('expanded_convbn', prefix=layerName))

    layer = DepthwiseConv2D(filters, strides=strides, depth_multiplier=1, padding='same', name=layerName)(layer)
    layer = BatchNormalization(axis=3, name = getName('BatchNorm_0', prefix=layerName))(layer)
    if activation == 'leakyRelu':
        layer = LeakyReLU(alpha=0.1)(layer)
    elif activation is not None:
        layer = Activation(activation, name= getName('Activation_0', prefix=layerName))(layer)

    layer = Conv2D(filters, 1, strides=1, padding='same', name= getName('point_conv', prefix=layerName))(layer)
    layer = BatchNormalization(axis=3, name = getName('BatchNorm_0', prefix=layerName))(layer)
    layer = add([layer, inputs])
    return layer
    

def conv2dbn(layer, filters, kernel_size, strides=1, padding='same',  activation='leakyRelu', use_bias=False, layerName = ''):
    layer = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=layerName)(layer)
    if not use_bias:
        layer = BatchNormalization(axis=3, scale=False, name=getName('BatchNorm', prefix=layerName))(layer)
    if activation == 'leakyRelu':
        layer = LeakyReLU(alpha=0.1)(layer)
    elif activation is not None:
        layer = Activation(activation, name=getName('Activation', prefix=layerName))(layer)
    return layer


def conv2dTransbn(layer, filters, kernel_size, strides=1, padding='same', activation='leakyRelu', use_bias=False, layerName = ''):
    layer = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=layerName)(layer)
    if not use_bias:
        layer = BatchNormalization(axis=3, scale=False, name=getName('BatchNorm', prefix=layerName))(layer)
    if activation == 'leakyRelu':
        layer = LeakyReLU(alpha=0.1)(layer)
    elif activation is not None:
        layer = Activation(activation, name=getName('Activation', prefix=layerName))(layer)
    return layer


def getFirstBlock(inputLayer, layerName):
    # (224, 224, 3)
    layer = conv2dbn(inputLayer, 32, 5, strides=1, layerName=layerName + 'conv1_32_5_2')
    #layer = conv2dbn(layer, 32, 5, strides=1, layerName=layerName + 'conv2_32_5_1')
    layer = conv2dbn(layer, 64, 5, strides=1, layerName=layerName + 'conv3_64_5_1')
    layer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
    #(112, 112, 64)
    return layer
   

def getMiddleBlock(inputLayer, scale=1, filters=128):
    # 112, 112, 64
    layer = conv2dbn(inputLayer, filters, 5, strides=1, layerName='resize1_1')
    
    #56, 56, 64
    firstLayer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
    
    #56, 56, 64
    layer = getMidBlock(firstLayer, filters, "Mid1", 0.3)
    
    
    #56, 56, 64
    layer = getMidBlock(layer, filters, "Mid2", 0.6)
    
    #56, 56, 64
    #layer = getMidBlock(layer, filters, "Mid3", 0.6)
    
    #56, 56, 64
    layer = getMidBlock(layer, filters, "Mid3", 0.5)
    
    layer = Add(name="add_layer")([firstLayer, layer])
    return layer
  
def getMidBlock(inputLayer, filters, layerName, scale=1):
    firstLayer = conv2dbn(inputLayer, filters, 5, strides=1, layerName=layerName + 'conv1_1')
    layer = MaxPooling2D(pool_size=(2, 2), padding='same')(firstLayer)
    layer = conv2dbn(layer, filters * 2, 4, strides=1, layerName=layerName + 'conv1_2')
    layer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
    layer = conv2dbn(layer, filters * 4, 4, strides=1, layerName=layerName + 'conv1_3')
    layer = MaxPooling2D(pool_size=(2, 2), padding='same')(layer)
    
    layer = conv2dTransbn(layer, filters * 4, 4, strides=1, layerName=layerName + 'trans1_1')
    #layer = UpSampling2D((2, 2))(layer)
    layer = conv2dTransbn(layer, filters * 2, 4, strides=2, layerName=layerName + 'trans1_2')
    #layer = UpSampling2D((2, 2))(layer)
    layer = conv2dTransbn(layer, filters, 5, strides=2, layerName=layerName + 'trans1_3')
    #layer = UpSampling2D((2, 2))(layer)
    layer = conv2dTransbn(layer, filters, 5, strides=2, layerName=layerName + 'trans1_4')
    
    layer = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(layer)[1:],
               arguments={'scale': scale},
               name=getName('res_scale', prefix=layerName))([layer, inputLayer])
    layer = LeakyReLU(alpha=0.1, name=layerName + 'res_relu')(layer)
    return layer
    

def getFinalBlock(inputLayer, layerName):
    layer = conv2dTransbn(inputLayer, 64, 5, strides=1, layerName = layerName + 'conv1_32_5_2')
    layer = conv2dTransbn(layer, 32, 5, strides=1, layerName = layerName + 'conv2_32_5_1')
    layer = conv2dTransbn(layer, 3, 5, strides=1, layerName = layerName + 'conv3_64_5_1')
    return layer

def getModel(imgInput):
    #layer = Concatenate([imgInput, imgCInput, imgVInput], axis = -1)
    layer = getMiddleBlock(imgInput)
    layer = getFinalBlock(layer, 'final')
    return layer
    

def myGenerator(inputLayer):
    imgoutput = getFirstBlock(inputLayer, 'first')
    modelOutPut = getModel(imgoutput)
    generator = Model(inputs= inputLayer, outputs=modelOutPut)
    generator.summary()
    return generator, modelOutPut