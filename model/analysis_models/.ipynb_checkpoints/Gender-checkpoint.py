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

    gender_model = Model(inputs=model.input, outputs=base_model_output)

    #--------------------------

    #load weights

    weight_path = './weights/gender_model_weights.h5'
    if os.path.isfile(weight_path) != True:
        print("gender_model_weights.h5 will be downloaded...")
        if not os.path.exists('./weights'):
            os.mkdir('./weights')

        url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'
        gdown.download(url, weight_path, quiet=False)

    gender_model.load_weights(weight_path)

    if feature_extracted:
        input_layer = Input((7,7,512))
        gender_h = gender_model.layers[-7](input_layer)
        gender_h = gender_model.layers[-6](gender_h)
        gender_h = gender_model.layers[-5](gender_h)
        gender_h = gender_model.layers[-4](gender_h)
        gender_h = gender_model.layers[-3](gender_h)
        gender_h = gender_model.layers[-2](gender_h)
        gender_out = gender_model.layers[-1](gender_h)
        gender_model = Model(inputs = input_layer, outputs=gender_out)
        return gender_model
    else:
        return gender_model

#--------------------------