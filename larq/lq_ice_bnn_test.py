#
# lq_ice_bnn_test.py --tfmodel models/lq_qat_bnn_1bit.tf --tflmodel tflite_models/lq_bnn_1bit.tflite
import larq_compute_engine as ice
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--tfmodel", required=True,
	help="path to pre-trained model")
ap.add_argument("-tm", "--tflmodel", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
im = Image.fromarray(test_images[0])
im.save("filename.bmp")

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = LabelEncoder().fit_transform(train_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

model = load_model(args["tfmodel"])
tflite_model = ice.convert_keras_model(model)
with open(args['tflmodel'], "wb") as f:
    f.write(tflite_model)

input_data = test_images[0]
label = test_labels[0]
print("ground truth label: {}", label)
x = tf.keras.preprocessing.image.img_to_array(input_data)
y = np.expand_dims(x, axis=0)
print("x.shape: {}", x.shape)
print("y.shape: {}", x.shape)
interpreter = ice.testing.Interpreter(tflite_model)
start = time.time()
pred = interpreter.predict(y, verbose=1)
end = time.time()
print("Elapsed time = {} ms", (end-start) * 1000)
print("pred.shape: {}", pred.shape)
print("pred: {}", pred)
print("Predicted label : {}", pred.argmax(1))