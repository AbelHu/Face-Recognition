import os

import cv2 as cv
import dlib
import imutils
from imutils import face_utils

if __name__ == '__main__':
    names = [f for f in os.listdir('data') if f.endswith('jpg')]
    detector = dlib.get_frontal_face_detector()
    results = []
    for name in names:
        print('processing ' + name)
        alias = name.split('.')[0]
        filename = os.path.join('data', name)
        image = cv.imread(filename)
        image = imutils.resize(image, width=500)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if (len(rects)):
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            image = image[y:y + h, x:x + w]
            image = cv.resize(image, (139, 139))
            cv.imwrite('data/{}.png'.format(alias), image)
