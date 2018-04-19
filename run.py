#! /usr/bin/env python3
import cv2
import sys
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from extraction.extractor import Extractor

extractor = Extractor()
cap = cv2.VideoCapture(0)

# load model
json_file = open('training/model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("training/model.h5")

def getLabel(image):
    image = np.array([[[i,i,i] for i in j] for j in image])

    # pre-process the image for classification
    image = cv2.resize(image, (24, 24))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    (closed, opened) = model.predict(image)[0]
    label = "Closed" if closed > opened else "Opened"
    return label

while(True):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # grayscale
    [selectedFace, eyes, box_config] = extractor.getFacialData(gray)
    if(len(selectedFace)==0 or len(eyes)!=2):
        continue

    # mark face
    x, y, w, h = selectedFace
    cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2) # draw rectangle
    # mark eyes
    roi_gray = gray[y:y+h, x:x+w]
    for(ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
    cv2.rectangle(gray, (box_config[0], box_config[1]),
        (box_config[0]+box_config[2], box_config[1]+box_config[3]),
        (0, 0, 255), 2)

    # print status
    eye_image1 = roi_gray[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
    eye_image2 = roi_gray[eyes[1][1]:eyes[1][1]+eyes[1][3], eyes[1][0]:eyes[1][0]+eyes[1][2]]
    print("left eye:", getLabel(eye_image1), "right eye:", getLabel(eye_image2))

    # display
    cv2.imshow('live', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit()

cap.release()
cv2.destroyAllWindows()
