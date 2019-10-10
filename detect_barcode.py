import numpy as np
import imutils
import cv2
import os, os.path

IN_PATH = "./in"
OUT_PATH = "./out"
VALID_IMAGE_EXTS = [".jpg", ".png", ]
WIDTH = 450
def detectBarcode(img_path):
  image = cv2.imread(IN_PATH + '/' + img_path)
  scale = WIDTH / image.shape[1]
  new_size = (WIDTH , int(scale * image.shape[1]))
  resized_image = cv2.resize(image, new_size)
  new_size = (WIDTH, int(scale * image.shape[1]))
  print(new_size)
  gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

  ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
  gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
  gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)

  blurred = cv2.blur(gradient, (11, 11))
  (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 6))
  closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

  closed = cv2.erode(closed, None, iterations = 4)
  closed = cv2.dilate(closed, None, iterations = 4)

  cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

  rect = cv2.minAreaRect(c)
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  box = box / scale
  box = box.astype(int)
  cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
  cv2.imwrite(OUT_PATH + '/out-' + img_path, image)

if __name__ == "__main__":
  for img_path in os.listdir(IN_PATH):
    ext = os.path.splitext(img_path)[1]
    if ext.lower() in VALID_IMAGE_EXTS:
      detectBarcode(img_path)