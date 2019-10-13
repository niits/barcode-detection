import numpy as np
import imutils
import cv2
import os
import os.path

IN_PATH = "./in"
OUT_PATH = "./out"

optimalValues = {
    1: {
        'name': '01.jpg',
        'gamma': 0,
        'blur': 11,
        'add': True,
        'threshold': 200,
        'kernel': {
                'type': cv2.MORPH_RECT,
                'x': 63,
                'y': 21
        },
        'erode': 10,
        'dilate': 10
    },
    2: {
        'name': '02.jpg',
        'gamma': 0,
        'blur': 11,
        'add': True,
        'threshold': 150,
        'kernel': {
                'type': cv2.MORPH_RECT,
                'x': 63,
                'y': 21
        },
        'erode': 10,
        'dilate': 10
    },
    3: {
        'name': '03.jpg',
        'gamma': 0.1,
        'blur': 11,
        'threshold': 150,
        'add': False,
        'kernel': {
                'type': cv2.MORPH_RECT,
                'x': 21,
                'y': 63
        },
        'erode': 10,
        'dilate': 10
    },
    4: {
        'name': '04.jpg',
        'gamma': 0,
        'blur': 11,
        'add': False,
        'threshold': 195,
        'kernel': {
                'type': cv2.MORPH_CROSS,
                'x': 63,
                'y': 21
        },
        'erode': 10,
        'dilate': 10
    },
    5: {
        'name': '05.jpg',
        'gamma': 0, 
        'blur': 11,
        'add': True,
        'threshold': 175,
        'kernel': {
                'type': cv2.MORPH_RECT,
                'x': 63,
                'y': 21
        },
        'erode': 10,
        'dilate': 10
    }
}

def detectBarcode(i):
    image = cv2.imread(IN_PATH + '/' + optimalValues[i]['name'])
    if optimalValues[i]['gamma'] != 0:
      table = np.array([((j / 255.0) **  optimalValues[i]['gamma']) * 255 for j in np.arange(0, 256)]).astype("uint8")
      image =  cv2.LUT(image, table)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    if optimalValues[i]['add']:
      gradient = cv2.add(gradX, gradY)
    else:
     gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (optimalValues[i]['blur'], optimalValues[i]['blur']))
    (_, thresh) = cv2.threshold(blurred, optimalValues[i]['threshold'], 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(optimalValues[i]['kernel']['type'], (optimalValues[i]['kernel']['x'], optimalValues[i]['kernel']['y']))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=optimalValues[i]['erode'])
    closed = cv2.dilate(closed, None, iterations=optimalValues[i]['dilate'])
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    image = cv2.imread(IN_PATH + '/' + optimalValues[i]['name'])
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imwrite(OUT_PATH + '/out-' + optimalValues[i]['name'], image)


if __name__ == "__main__":
    for index in optimalValues:
        detectBarcode(index)
