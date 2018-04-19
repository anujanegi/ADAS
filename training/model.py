from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.optimizers import Adam
from keras import backend as K

def def_model(height, width, depth) :

	#LetNet architecture
    model = Sequential()
    shape = (height, width, depth)

    #first set
    model.add(Conv2D(20, (5, 5), padding = "same", input_shape = shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

    #second set
    model.add(Conv2D(50, (5, 5), padding = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

    #connecting
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(2, activation = "sigmoid"))

    return model

def compile_model(model):
    epoch = 25
    initialRate = 1e-3
    Optimizer = Adam(lr = initialRate, decay = initialRate/epoch)
    model.compile(loss = 'binary_crossentropy', optimizer = Optimizer, metrics=['accuracy'])
    return model

def fit_model(model, trainX, trainY, testX, testY):
    augment = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    size = 50
    EPOCHS = 20
    step = len(trainX)//size

    model.fit_generator(augment.flow(trainX, trainY, batch_size = size), validation_data=(testX, testY), steps_per_epoch = step, epochs = EPOCHS, verbose=1)
    return model

def evaluate(model, X, Y):
	scores = model.evaluate(X,Y)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
