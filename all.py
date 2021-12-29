import os
import cv2
import random
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from keras.models import Sequential
from tensorflow.keras.utils import to_categorical, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, regularizers, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import service as s

####################################################################################3

img_size = 96

###########################################################################################

Train_path = "dataset/Train/Train-"
Test_path = "dataset/Test"

Easy_data = s.load_data(Train_path+'Easy', train=True , img_size=img_size)
Medium_data = s.load_data(Train_path+'Medium', train=True , img_size=img_size)
Hard_data = s.load_data(Train_path+'Hard', train=True , img_size=img_size)
Real_data = s.load_data(Test_path, train=False)

data = np.concatenate([Easy_data,Medium_data, Hard_data], axis=0) # ,Medium_data, Hard_data

del Easy_data , Medium_data, Hard_data

###########################################################################################

X_train , y_train = s.feature_label(data)

X_train = np.array(X_train).reshape(-1, img_size, img_size, 1)
X_train = X_train / 255.0 # Normalize to [0, 1]
y_train = to_categorical(y_train, num_classes=10) # 10 fingers per person
X_train_train, X_test_validate, y_train_train, y_test_validate = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1)


##############################################################################################

X_test , y_test = s.feature_label(Real_data)

X_test = np.array(X_test).reshape(-1, img_size, img_size, 1)
X_test = X_test / 255.0
y_test = to_categorical(y_test, num_classes=10)

##############################################################################################


print("Shapes:                  Feature shape    label shape")
print("----------------------------------------------------")
print("full finger data:  ", X_train.shape, y_train.shape)
print("finger_Train:      ", X_train_train.shape, y_train_train.shape)
print("finger_Validation: ", X_test_validate.shape, y_test_validate.shape)
print("finger_Test:       ", X_test.shape, y_test.shape)

del data, Real_data , X_train , y_train

##############################################################################################

final_Dense_units = 10  #### 10 classes

##################################### MODEL ANN ######################################


model_name_ann = 'FingerPrint_Model_ANN'
model_ann = Sequential(name=model_name_ann)
model_ann.add(layers.Dense(64, input_shape = (96, 96, 1), activation='relu')) #512
model_ann.add(layers.Flatten())
model_ann.add(layers.Dense(32, activation='relu')) # 256
model_ann.add(layers.Dense(final_Dense_units, activation='softmax'))

model_ann.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
model_ann.summary()

##############################################################################################

plot_model(model_ann , show_shapes=True, to_file='./FingerPrint_Model_ANN.png')

###############################################   CNN ########################################


model_name_cnn = 'FingerPrint_Model_CNN'

model_cnn = Sequential(name=model_name_cnn)
model_cnn.add(layers.Conv2D(32, (5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape = (96, 96, 1)))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(64,(5, 5), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Conv2D(128,(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model_cnn.add(layers.BatchNormalization())
model_cnn.add(layers.MaxPool2D((2, 2)))
model_cnn.add(layers.Dropout(0.3))
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(256, activation='relu'))
model_cnn.add(layers.Dropout(0.4))
model_cnn.add(layers.Dense(final_Dense_units, activation='softmax'))

model_cnn.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
model_cnn.summary()


##############################################################################################

plot_model(model_cnn, show_shapes=True, to_file='./FingerPrint_Model_CNN.png')

#############################################################################################

ReduceLR_minlr = 1e-7
epochs = 1                      #20
batch_size = 64

####################################################################################

####################### CALLBACK ANN
CallBack_ann = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.1, patience=1, min_lr=ReduceLR_minlr, verbose=1),
    callbacks.TensorBoard(log_dir="./log_dir/ANN/"+model_name_ann)]

####################### CALLBACK CNN

CallBack_cnn = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(factor=0.1, patience=1, min_lr=ReduceLR_minlr, verbose=1),
    callbacks.TensorBoard(log_dir="./log_dir/CNN/"+model_name_cnn)]


####################################################################################

######################## FITTING MODEL ANN 
history_ann = model_ann.fit(X_train_train, y_train_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    validation_data = (X_test_validate, y_test_validate),
                    verbose = 1, callbacks= CallBack_ann)

######################## FITTING MODEL CNN

history_cnn = model_cnn.fit(X_train_train, y_train_train,
                    batch_size = batch_size,
                    epochs = epochs, 
                    validation_data = (X_test_validate, y_test_validate),
                    verbose = 1, callbacks= CallBack_cnn)


#############################################################################################

del X_train_train, X_test_validate, y_train_train, y_test_validate

###############################################################################################



##################### ANN

acc_ann , validate_acc_ann , loss_ann , validate_loss_ann = s.model_stat(history_ann)

s.draw_chart_train_validate(acc=acc_ann , model_name=model_name_ann , validate_acc=validate_acc_ann)

s.draw_chart_loss_validate(acc=acc_ann , model_name=model_name_ann , loss=loss_ann, validate_loss=validate_acc_ann)

####################  CNN
acc_cnn , validate_acc_cnn , loss_cnn , validate_loss_cnn = s.model_stat(history_cnn)

s.draw_chart_train_validate(acc=acc_cnn , model_name=model_name_cnn , validate_acc=validate_acc_cnn)

s.draw_chart_loss_validate(acc=acc_cnn , model_name=model_name_cnn , loss=loss_cnn, validate_loss=validate_acc_cnn)

###########################################################################################

####################### evaluate ANN

testing_acc_ann = model_ann.evaluate([X_test], [y_test], verbose=0)
print("FingerPrint recognition accuracy: ",testing_acc_ann[1]*100, "%")

####################### evaluate CNN

testing_acc_cnn = model_cnn.evaluate([X_test], [y_test], verbose=0)
print("FingerPrint recognition accuracy: ",testing_acc_cnn[1]*100, "%")

#################################################################################################

######################### confusion matrix ANN 

s.confusion_matrix_model(X_test , model_ann , y_test , model_name=model_name_ann)

######################### confusion matrix CNN

s.confusion_matrix_model(X_test , model_cnn , y_test , model_name=model_name_cnn)

############################################################################################

######################### SAVE ANN 

model_ann.save(model_name_ann+'.h5')

######################### SAVE CNN 

model_cnn.save(model_name_cnn+'.h5')

###############################################################################################
