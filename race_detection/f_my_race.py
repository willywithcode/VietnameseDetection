
import os
from pathlib import Path
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import face_recognition


class Race_Model():
    def __init__(self):
        self.model = self.loadModel()
        self.race_labels = ['foreigner','vietnamese']

    def predict_race(self,face_image):
        image_preprocesing = self.transform_face_array2race_face(face_image)
        race_predictions = self.model.predict(image_preprocesing)
        print(race_predictions)
        # result_race = self.race_labels[np.argmax(race_predictions)]
        if(race_predictions[0][1]>race_predictions[0][0]):
            result_race = self.race_labels[0]
        else:
            result_race = self.race_labels[1]
        return result_race

    def loadModel(self): 
        home = str(Path.cwd()) 
        
        race_model = load_model(home+'/weights/vietname_detection_9_model.h5')
        return race_model
        #--------------------------
    def transform_face_array2race_face(self,face_array,target_size = (128, 128)):
        detected_face = face_array
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        img_pixels /= 255
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        
        return img_pixels