import os
import cv2
import keras
import numpy as np
import random

UPLOAD_FOLDER = 'static/uploads/'

def show_fingername(fingernum):
    if fingernum>=5:
        fingername = "right "
        fingernum -= 5
    else: fingername = "left "
    if fingernum==0:
        fingername += "little"
    elif fingernum==1:
        fingername += "ring"
    elif fingernum==2:
        fingername += "middle"
    elif fingernum==3:
        fingername += "index"
    else: fingername += "thumb"
    return fingername

def prediction(image , model=''):
    loaded_model = keras.models.load_model("FingerPrint_Model_ANN.h5") if model == 'ANN' else keras.models.load_model("FingerPrint_Model_CNN.h5")
    
    image = UPLOAD_FOLDER + image
    img_size = 96
    img_array = cv2.imread(image, cv2.IMREAD_GRAYSCALE) # 'dataset/Train/Train-Easy/1__M_Left_thumb_finger_Zcut.BMP'
    img_array = cv2.resize(img_array, (img_size, img_size))
    img_array = np.array(img_array).reshape(-1, img_size, img_size, 1)
    p = loaded_model.predict(img_array)
    max_value = np.argmax(p)
    predict = show_fingername(max_value)
    expected = [show_fingername(i) for i in range(10)]
    predict_proba = dict(zip(expected, [t for t in p[0]]))
    return predict , predict_proba