from . import VGGFace
import os
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation, Input

def loadModel(feature_extracted=False):

    model = VGGFace.baseModel()

    #--------------------------

    classes = 2
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    #--------------------------

    mask_model = Model(inputs=model.input, outputs=base_model_output)

    #--------------------------

    #load weights

    weight_path = './weights/mask_model_weights.h5'
    mask_model.load_weights(weight_path)

    if feature_extracted:
        input_layer = Input((7,7,512))
        mask_h = mask_model.layers[-7](input_layer)
        mask_h = mask_model.layers[-6](mask_h)
        mask_h = mask_model.layers[-5](mask_h)
        mask_h = mask_model.layers[-4](mask_h)
        mask_h = mask_model.layers[-3](mask_h)
        mask_h = mask_model.layers[-2](mask_h)
        mask_out = mask_model.layers[-1](mask_h)
        mask_model = Model(inputs = input_layer, outputs=mask_out)
        return mask_model
    else:
        return mask_model

#--------------------------