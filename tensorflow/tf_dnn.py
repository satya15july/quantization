# Usage: tf_ptq_dnn.py --tfmodel models/baseline_dnn_32bit.tf
import tensorflow as tf
import larq as lq
from sklearn.preprocessing import LabelEncoder
import time
import argparse
import numpy as np
import tempfile
import os
from simple_network import MyNet

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--tfmodel", required=True,
	help="path to output model")

args = vars(ap.parse_args())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = LabelEncoder().fit_transform(train_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

# Normalize pixel values to be between -1 and 1
#train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
x = MyNet(classes = 10)(input_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=x)

model.summary()
lq.models.summary(model)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=6)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy {test_acc * 100:.2f} %")

print("[INFO] serializing model...")
model.save(args["tfmodel"])