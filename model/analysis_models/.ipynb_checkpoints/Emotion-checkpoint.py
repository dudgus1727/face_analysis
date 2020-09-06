import os
import gdown
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
import zipfile

def loadModel():
	
	num_classes = 7
	
	model = Sequential()

	#1st convolution layer
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	#2nd convolution layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())

	#fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='softmax'))
	
	#----------------------------
	
	weight_path = './weights/facial_expression_model_weights.h5'
	if os.path.isfile(weight_path) != True:
		print("facial_expression_model_weights.h5 will be downloaded...")
		if not os.path.exists('./weights'):
			os.mkdir('./weights')
		
		#zip
		url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
		output = './weights/facial_expression_model_weights.zip'
		gdown.download(url, output, quiet=False)
		
		#unzip facial_expression_model_weights.zip
		with zipfile.ZipFile(output, 'r') as zip_ref:
			zip_ref.extractall('./weights/')
		
	model.load_weights(weight_path)
	
	return model
	
	#----------------------------
	
	return 0