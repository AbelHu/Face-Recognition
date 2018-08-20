import json

import cv2 as cv
import dlib
import imutils
import keras.backend as K
import numpy as np
from imutils import face_utils
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import load_model

if __name__ == '__main__':
    img_size = 139
    model = load_model('models/model.10-0.0156.hdf5')
    detector = dlib.get_frontal_face_detector()
    image_inputs = np.empty((1, img_size, img_size, 3), dtype=np.float32)
    dummy_input = np.zeros((1, img_size, img_size, 3), dtype=np.float32)

    filename = 'images/foamliu.png'
    image = cv.imread(filename)
    image = imutils.resize(image, width=500)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    image = image[y:y + h, x:x + w]
    image = cv.resize(image, (img_size, img_size))
    image = image[:, :, ::-1].astype(np.float32)
    image_inputs[0] = preprocess_input(image)
    y_pred = model.predict([image_inputs, dummy_input, dummy_input])
    e1 = y_pred[0, 0:128]

    with open('data/results.json', 'r') as file:
        embeddings = json.load(file)

    distances = []
    for e in embeddings:
        e2 = e['embedding']
        distance = np.linalg.norm(e1 - e2) ** 2
        distances.append(distance)

    index = np.argmin(distances)
    print(embeddings[index]['alias'])

    for i in range(len(embeddings)):
        print('alias: {} distance:{}'.format(embeddings[i]['alias'], distances[i]))

    K.clear_session()
