import numpy as np
import pickle
import cv2
import os
import json
import sys
from model.analysis_models import Age, Emotion, VGGFace, Gender, Mask
from keras.models import Model

feature_extractor = VGGFace.loadModel(feature_extractor=True)

age_no_base = Age.loadModel(feature_extracted = True)
age_predict = age_no_base(feature_extractor.output)
gender_no_base = Gender.loadModel(feature_extracted = True)
gender_predict = gender_no_base(feature_extractor.output)

model = Model(inputs=feature_extractor.input, outputs=[gender_predict, age_predict])

emotion_model = Emotion.loadModel()
mask_model = Mask.loadModel(feature_extracted = False)

face_detecter_path = 'model/face_detector/'
protoPath = face_detecter_path + "deploy.prototxt"
modelPath = face_detecter_path + "face_detect.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

w, h = (640, 480)
size_arr = np.array([w, h, w, h])
confidence =0.7

emotion_labels = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
emotion_color = [(0,0,255), (0,69,255), (128,0,128), (147,20,255), (255,191,0), (0,215,255), (50,205,50)]
age_indexes = np.arange(101)
gender_labels = np.array(['Woman', 'Man'])
race_labels = np.array(['Asian', 'Indian', 'Black', 'White', 'Middle eastern', 'Latino hispanic'])

# font = cv2.FONT_HERSHEY_TRIPLEX
font = 0

# cap = cv2.VideoCapture('test.mp4')
cap = cv2.VideoCapture(0)
cap.set(3,w);
cap.set(4,h);

while True:
    ret, image = cap.read()
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward().squeeze()
    detections = detections[detections[:, 2]>confidence]
    detections[detections[:, 3] <0, 3] = 0
    detections[detections[:, 4] <0, 4] = 0
    detections = detections[(detections[:, 5] -detections[:, 3]) > 0.05]  #filtering by face width size
    detections = detections[(detections[:, 6] -detections[:, 4]) > 0.05]  #filtering by face height size

    if len(detections) > 0:
        faces_check_mask =[]
        detections[:,3:7] *= size_arr
        for detection in detections:
            (startX, startY, endX, endY) = detection[3:7].astype("int")
            face=image[startY:endY,startX:endX]
            faces_check_mask.append(face)

        inputs = np.empty((len(faces_check_mask), 224, 224, 3), dtype='float32')
        for i, face in enumerate(faces_check_mask):
            inputs[i] = (cv2.resize(face, (224,224))/255).astype('float32')
#         mask_predictions, gender_predictions, age_predictions = model.predict_on_batch(inputs)
        mask_predictions = mask_model.predict(inputs)
        mask_predictions = np.argmax(mask_predictions, axis =-1)
        
        ext_w = (detections[:, 5:6] -detections[:, 3:4])*0.05
        ext_h = (detections[:, 6:7] -detections[:, 4:5])*0.05
        # ext_w = (detections[:, 5:6] -detections[:, 3:4])*0.1
        # ext_h = (detections[:, 6:7] -detections[:, 4:5])*0.1
        detections[:, 3:4] -= ext_w
        detections[:, 4:5] -= ext_h
        detections[:, 5:6] += ext_w
        detections[:, 6:7] += ext_h
        unmask_detection = detections[mask_predictions==1]
        unmask_detection[unmask_detection[:, 3] <0, 3] = 0
        unmask_detection[unmask_detection[:, 4] <0, 4] = 0
        dunmask_etection = unmask_detection[((unmask_detection[:, 5] -unmask_detection[:, 3]))/w > 0.05]  #filtering by face width size
        dunmask_etection = unmask_detection[((unmask_detection[:, 6] -unmask_detection[:, 4]))/h > 0.05]  #filtering by face height size
        if len(unmask_detection)>0:
            faces_unmask =[]
            del_list=[]
            for i, detection in enumerate(unmask_detection):
                (startX, startY, endX, endY) = detection[3:7].astype("int")
                face=image[startY:endY,startX:endX]
                faces_unmask.append(face)
                
                
            print(faces_unmask) 
            inputs = np.empty((len(faces_unmask), 224, 224, 3), dtype='float32')
            for i, face in enumerate(faces_unmask):
                inputs[i] = (cv2.resize(face, (224,224))/255).astype('float32')
            gender_predictions, age_predictions = model.predict(inputs)

            inputs = np.empty((len(faces_unmask), 48, 48,1), dtype='float32')
            for i, face in enumerate(faces_unmask):
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
                gray = cv2.resize(gray, (48,48))/255
                inputs[i] = gray[:,:,np.newaxis].astype('float32')
            emotion_predictions = emotion_model.predict(inputs)
            emotion_predictions /= emotion_predictions.sum(axis=1)[:,np.newaxis]
            # age_predictions  = np.sum(age_predictions*age_indexes, axis =1)
            age_predictions  = np.argmax(age_predictions, axis =-1)
            gender_predictions = np.argmax(gender_predictions, axis =-1)
            gender_predictions = gender_labels[gender_predictions]

        mask_rect  = detections[mask_predictions==0][:,3:7].astype('int')
        for startX, startY, endX, endY in mask_rect:
            cv2.rectangle(image, (startX, startY), (endX, endY), (255,0,0), 3)
            interval = (endY-startY)//18
            startY -= int(interval*0.5)
            cv2.putText(image, "masked",
                        (startX, startY),fontFace=font, fontScale=interval*0.060, color=(255,0,0), thickness=2)

        if len(unmask_detection)>0:
            unmask_rect  = unmask_detection[:,3:7].astype('int')
            for i, (startX, startY, endX, endY) in enumerate(unmask_rect):
                cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 3)

                interval = (endY-startY)//18
                endY += int(interval*0.5)

                for j, prop in enumerate(emotion_predictions[i]):
                    startY_ = endY + int(interval*1.1*j)
                    endY_ = startY_ + int(interval*0.9)
                    cv2.rectangle(image, (startX, startY_),(endX, endY_), emotion_color[j], 2)
                    cv2.rectangle(image, (startX, startY_),(int((endX-startX)*prop)+startX, endY_), emotion_color[j], -1)
                    cv2.putText(image,
                                '{:8s} : {:.2f}%'.format(emotion_labels[j],prop*100), (startX+4, endY_-2), fontFace=font, fontScale=interval*0.028, color=(0,0,0), thickness=2)

                startY -= int(interval*0.5)
                endX += int(interval*0.5)
                cv2.putText(image, "without mask",
                            (startX, startY),fontFace=font, fontScale=interval*0.060, color=(0,0,255), thickness=2)
                cv2.putText(image, "Age     : {}".format(int(age_predictions[i])),
                            (endX, startY+interval*4),fontFace=font, fontScale=interval*0.040, color=(255,0,0), thickness=2)
                cv2.putText(image, "Gender  : {}".format(gender_predictions[i]),
                            (endX, startY+interval*2),fontFace=font, fontScale=interval*0.040, color=(255,0,0), thickness=2)

    cv2.imshow('Face Analysis', cv2.resize(image,(w,h)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()