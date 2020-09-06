from . import VGGFace
import os
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation, Input
import zipfile

def loadModel(feature_extracted=False):

    model = VGGFace.baseModel()

    #--------------------------

    classes = 6
    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    #--------------------------

    race_model = Model(inputs=model.input, outputs=base_model_output)

    #--------------------------

    #load weights
    weight_path = './weights/race_model_single_batch.h5'
    if os.path.isfile(weight_path) != True:
        print("race_model_single_batch.h5 will be downloaded...")
        if not os.path.exists('./weights'):
            os.mkdir('./weights')
        #zip
        url = 'https://drive.google.com/uc?id=1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj'
        output = './weights/race_model_single_batch.zip'
        gdown.download(url, output, quiet=False)

        #unzip race_model_single_batch.zip
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('./weights/')

    race_model.load_weights(weight_path)

    if feature_extracted:
        input_layer = Input((7,7,512))
        race_h = race_model.layers[-7](input_layer)
        race_h = race_model.layers[-6](race_h)
        race_h = race_model.layers[-5](race_h)
        race_h = race_model.layers[-4](race_h)
        race_h = race_model.layers[-3](race_h)
        race_h = race_model.layers[-2](race_h)
        race_out = race_model.layers[-1](race_h)
        race_model = Model(inputs = input_layer, outputs=race_out)
        return race_model
    else:
        return race_model

#--------------------------
