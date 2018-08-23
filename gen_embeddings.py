import json
import os

import cv2 as cv
import numpy as np
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import load_model

if __name__ == '__main__':
    img_size = 139
    model = load_model('models/model.10-0.0156.hdf5')

    names = [f for f in os.listdir('data') if f.endswith('png')]
    dummy_input = np.zeros((1, img_size, img_size, 3), dtype=np.float32)

    results = []
    for name in names:
        print('processing ' + name)
        alias = name.split('.')[0]
        filename = os.path.join('data', name)
        image_inputs = np.empty((1, img_size, img_size, 3), dtype=np.float32)
        image_bgr = cv.imread(filename)
        image_bgr = cv.resize(image_bgr, (img_size, img_size), cv.INTER_CUBIC)
        image_rgb = image_bgr[:, :, ::-1].astype(np.float32)
        image_inputs[0] = preprocess_input(image_rgb)
        y_pred = model.predict([image_inputs, dummy_input, dummy_input])
        embedding = y_pred[0, 0:128]

        results.append({'alias': alias, 'embedding': embedding.tolist()})

    with open('data/results.json', 'w') as file:
        json.dump(results, file, indent=4)
