# lq_bnn_network.py --model models/lq_qat_bnn_1bit.tf
import tensorflow as tf
import larq as lq
from sklearn.preprocessing import LabelEncoder
import time
import argparse
import larq_compute_engine as ice
import tempfile
import os
import pathlib
from simple_bnn_network import MyBnnNet

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to save larq model")

#ap.add_argument("-l", "--lqmodel", required=True,
#	help="path to larq-tflite model")

args = vars(ap.parse_args())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = LabelEncoder().fit_transform(train_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
x = MyBnnNet(classes = 10) (input_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=x)

model.summary()
lq.models.summary(model)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=6)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"BNN Test accuracy {test_acc * 100:.2f} %")

print("[INFO] serializing model...")
model.save(args["model"])

tflite_model_binary = ice.convert_keras_model(model)

tflite_models_dir = pathlib.Path("tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_binary_file = tflite_models_dir/"lq_qat_bnn_1bit.tflite"
tflite_model_binary_file.write_bytes(tflite_model_binary)

_, binarized_file = tempfile.mkstemp('.tflite')

with open(binarized_file, 'wb') as f:
  f.write(tflite_model_binary)

print("Binarized model in Mb:", os.path.getsize(binarized_file) / float(2**20))