from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import random
import cv2
import os

def load_data():
    X = [] #data
    Y = [] #label

    dataset = "dataset"
    path = sorted(list(paths.list_images(dataset)))
    random.seed(7)
    random.shuffle(path)

    for image_path in path:
        image = cv2.imread(image_path)
        image = img_to_array(image)
        X.append(image)
        label = image_path.split(os.path.sep)[-2]
        if label == "closedEyes" :
            label = 0
        else:
            label = 1
        Y.append(label)

    #scaling
    X = np.array(X, dtype = "float")/255.0
    Y = np.array(Y)
    (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.25, random_state=10)
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)
    return trainX, trainY, testX, testY
