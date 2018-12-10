import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras import backend as K
from matplotlib import pyplot as plt
from IPython.display import clear_output

def def_model(height, width, depth) :

    model = Sequential()
    shape = (height, width, depth)

    model = Sequential()
    model.add(Conv2D(50, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    return model

def compile_model(model):
    epoch = 100
    initialRate = 1e-3
    Optimizer = Adam(lr = initialRate, decay = initialRate/epoch)
    model.compile(loss = 'binary_crossentropy', optimizer = Optimizer, metrics=['accuracy'])
    return model

def fit_model(model, trainX, trainY, testX, testY):
    augment = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    size = 50
    EPOCHS = 100
    step = len(trainX)//size

    model.fit_generator(augment.flow(trainX, trainY, batch_size = size), validation_data=(testX, testY), steps_per_epoch = step, epochs = EPOCHS, verbose=1)
    return model

def evaluate(model, X, Y):
	scores = model.evaluate(X,Y)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
