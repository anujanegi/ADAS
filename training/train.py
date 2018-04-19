import pickle
from keras.models import model_from_json
from data import *
from model import *

trainX, trainY, testX, testY = load_data()
model = fit_model(compile_model(def_model(24, 24, 3)), trainX, trainY, testX, testY)
evaluate(model, trainX, trainY)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
