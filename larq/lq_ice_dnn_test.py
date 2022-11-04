# optimized_bnn_test.py --model models/lq_dnn_model --tfmodel models/tfbnn_model
import larq_compute_engine as ice
import tensorflow as tf
import argparse
from tensorflow.keras.models import load_model
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-tm", "--tfmodel", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = LabelEncoder().fit_transform(train_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

model = load_model(args["model"])
tflite_model = ice.convert_keras_model(model)
with open(args['tfmodel'], "wb") as f:
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