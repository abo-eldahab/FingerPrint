import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

def load_data(path, train=True , img_size=96):
    print("Loading data from: ", path)
    data = []
    for img in os.listdir(path):
        imgname, ext = os.path.splitext(img)
        ID, etc = imgname.split('__')
        ID = int(ID) - 1 # to_categorical encodes starting from 0
        if train:
            _, lr, finger, _, _ = etc.split('_')
        else:
            _, lr, finger, _  = etc.split('_')
        if lr=='Left':
            base = 0 # left hand corresponding to 0-4
        else: base  = 5 # right hand corresponding to 5-9
        if finger=="little":
            fingerNum = base + 0
        elif finger=='ring':
            fingerNum = base + 1
        elif finger=='middle':
            fingerNum = base + 2
        elif finger=='index':
            fingerNum = base + 3 
        else: fingerNum = base + 4
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img_array, (img_size, img_size))
        data.append([ID, fingerNum, img_resize])
    return data

def feature_label(data):
    X_feature , y_finger = [], []
    for SubjectID, fingerNum, feature in data:
        X_feature.append(feature)                          ###### IMAGE
        y_finger.append(fingerNum)              ###### 0 ----- >> 9
    return X_feature , y_finger

def model_stat(model_history):
    acc_ann = model_history.history['accuracy']
    validate_acc_ann = model_history.history['val_accuracy']
    loss_ann = model_history.history['loss']
    validate_loss_ann = model_history.history['val_loss']

    return acc_ann , validate_acc_ann , loss_ann , validate_loss_ann

def draw_chart_train_validate(acc , model_name , validate_acc):

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(10,10))
    plt.plot(epochs, acc, label='Training acc of '+model_name)
    plt.plot(epochs, validate_acc, label='Validation acc of '+model_name)
    plt.title('Training and validation accuracy of '+model_name)
    plt.legend()
    # plt.show()
    plt.savefig(model_name+'_accuracy.png')

def draw_chart_loss_validate(acc , model_name , loss , validate_loss):

    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(10,10))
    plt.plot(epochs, loss,  label='Training loss of '+model_name)
    plt.plot(epochs, validate_loss, label='Validation loss of '+model_name)
    plt.title('Training and validation loss of '+model_name)
    plt.legend()
    # plt.show()
    plt.savefig(model_name+'_loss.png')

def plot_confusion_matrix(conmat, classes, model_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(conmat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        conmat = conmat.astype('float') / conmat.sum(axis=1)[:, np.newaxis]

    thresh = conmat.max() / 2.
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        plt.text(j, i, conmat[i, j],
                 horizontalalignment="center",
                 color="white" if conmat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Real label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(model_name+'.png')


def confusion_matrix_model(X_test , model , y_test , model_name):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1) 
    confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = range(10) , model_name=model_name)
