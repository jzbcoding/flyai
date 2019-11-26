from PIL import Image
import numpy
from path import DATA_PATH
import os

# image = Image.open(path)
# image = image.convert("RGB")
# image = image.resize((64, 64), Image.BILINEAR)
# x_data = numpy.array(image)
# image.show()

import cv2
path = os.path.join(DATA_PATH, 'img', '32.tif')
img = cv2.imread(path)
print(img.shape)
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
cv2.imshow('result.jpg',img)
cv2.waitKey(0)