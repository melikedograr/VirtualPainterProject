import cv2
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('./ShapeDetectionDeepLearning/shape_detection_model')
canvas = cv2.imread('image.jpg')
gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
print('canvas :', canvas.shape)
print('gray :', gray.shape)

resized = cv2.resize(gray, (200,200))
print('resized :', resized.shape)

images = [resized]
images = np.array(images)
prediction = model.predict(images)
print(prediction)
