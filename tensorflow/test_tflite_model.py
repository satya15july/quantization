# run tf_ptq_dnn.py by invoking tf_ptq_dnn.py --model models/tf_dnn_model
# load the model by invoking
# tflite_dnn_quant_test.py --tfmodel tflite_models/pqt_dnn_32bit.tflite --tfqmodel tflite_models/pqt_dnn_8bit.tflite
#
import tensorflow as tf
import larq as lq
from sklearn.preprocessing import LabelEncoder
import time
import argparse
import numpy as np
import larq as lq
from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-tfm", "--tfmodel", required=True,
	help="path to pre-trained 32 bit model")
ap.add_argument("-tfqm", "--tfqmodel", required=True,
	help="path to pre-trained 8 bit model")
args = vars(ap.parse_args())

tflite_model_file = args["tfmodel"]
tflite_model_quant_file = args["tfqmodel"]
print("tflite_model_file: {}", tflite_model_file)
print("tflite_model_quant_file: {}", tflite_model_quant_file)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = LabelEncoder().fit_transform(train_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

# Normalize pixel values to be between -1 and 1
#train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

def run_tflite_model(tflite_file, test_image_indices):
  global test_images
  print("run_tflite_model,  test_image_indices : {}", test_image_indices)
  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  print("run_tflite_model,  predictions.shape : {}", predictions.shape)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    #print("run_tflite_model,  test_image.shape : {}", test_image.shape)
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions

import matplotlib.pylab as plt
import time

# Change this to test a different image
test_image_index = 1

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
  print("test_image_index : {}", test_image_index)
  global test_labels
  start = time.time()
  predictions = run_tflite_model(tflite_file, [test_image_index])
  end = time.time()
  print("Evaluation Time for model type: {} = {} ms ", model_type, (end-start) * 1000)
  print("run_tflite_model,  predictions.shape : {}", predictions.shape)
  plt.imshow(test_images[test_image_index])
  template = model_type + " Model \n True:{true}, Predicted:{predict}"
  _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
  plt.grid(False)


test_model(tflite_model_file, test_image_index, model_type="Float")

test_model(tflite_model_quant_file, test_image_index, model_type="Quantized")

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global test_images
  global test_labels
  print("evaluate_model model_type: {}", model_type)
  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))


evaluate_model(tflite_model_file, model_type="Float")

evaluate_model(tflite_model_quant_file, model_type="Quantized")
