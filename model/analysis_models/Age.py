from . import VGGFace
import os
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation, Input

def loadModel(feature_extracted = False):

    model = VGGFace.baseModel()

    #--------------------------

    classes = 101
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    #--------------------------

    age_model = Model(inputs=model.input, outputs=base_model_output)

    #--------------------------

    #load weights

    weight_path ='./weights/age_model_weights.h5'
    if os.path.isfile(weight_path) != True:
        print("age_model_weights.h5 will be downloaded...")
        if not os.path.exists('./weights'):
            os.mkdir('./weights')

        url = 'https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV'
        gdown.download(url, weight_path, quiet=False)

    age_model.load_weights(weight_path)

    if feature_extracted:
        input_layer = Input((7,7,512))
        age_h = age_model.layers[-7](input_layer)
        age_h = age_model.layers[-6](age_h)
        age_h = age_model.layers[-5](age_h)
        age_h = age_model.layers[-4](age_h)
        age_h = age_model.layers[-3](age_h)
        age_h = age_model.layers[-2](age_h)
        age_out = age_model.layers[-1](age_h)
        age_model = Model(inputs = input_layer, outputs=age_out)
        return age_model
    else:
        return age_model