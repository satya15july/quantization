# tf_qat_dnn.py --model models/qat_dnn_8bit.tf
import tensorflow as tf
import larq as lq
from sklearn.preprocessing import LabelEncoder
import time
import argparse
import numpy as np
import tensorflow_model_optimization as tfmot
import os
import tempfile
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_labels = LabelEncoder().fit_transform(train_labels)
test_labels = LabelEncoder().fit_transform(test_labels)

# Normalize pixel values to be between -1 and 1
# train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


class DefaultConv2DQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # List all of your weights
    weights = {
        "kernel": LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)
    }

    # List of all your activations
    activations = {
        "activation": MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)
    }

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        # return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]
        output = []
        for attribute, quantizer in self.weights.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))

        return output

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        # return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]
        output = []
        for attribute, quantizer in self.activations.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))

        return output

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        # layer.kernel = quantize_weights[0]
        count = 0
        for attribute in self.weights.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_weights[count])
                count += 1

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        # layer.activation = quantize_activations[0]
        count = 0
        for attribute in self.activations.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_activations[count])
                count += 1

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope


class Custom2DLayer(tf.keras.layers.Conv2D):
    pass


class CustomDenseLayer(tf.keras.layers.Dense):
    pass


quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
q_aware_model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
q_aware_model.add(quantize_annotate_layer(Custom2DLayer(32, (3, 3),
                                                        use_bias=False,
                                                        input_shape=(28, 28, 1)),
                                          DefaultConv2DQuantizeConfig()))
q_aware_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
q_aware_model.add(tf.keras.layers.BatchNormalization(scale=False))

q_aware_model.add(quantize_annotate_layer(Custom2DLayer(64, (3, 3), use_bias=False), DefaultConv2DQuantizeConfig()))
q_aware_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
q_aware_model.add(tf.keras.layers.BatchNormalization(scale=False))

q_aware_model.add(quantize_annotate_layer(Custom2DLayer(64, (3, 3), use_bias=False), DefaultConv2DQuantizeConfig()))
q_aware_model.add(tf.keras.layers.BatchNormalization(scale=False))
q_aware_model.add(tf.keras.layers.Flatten())

q_aware_model.add(quantize_annotate_layer(CustomDenseLayer(64, use_bias=False), DefaultConv2DQuantizeConfig()))
q_aware_model.add(tf.keras.layers.BatchNormalization(scale=False))
q_aware_model.add(quantize_annotate_layer(CustomDenseLayer(10, use_bias=False), DefaultConv2DQuantizeConfig()))
q_aware_model.add(tf.keras.layers.BatchNormalization(scale=False))
q_aware_model.add(tf.keras.layers.Activation("softmax"))

with quantize_scope(
        {'DefaultConv2DQuantizeConfig': DefaultConv2DQuantizeConfig,
         'Custom2DLayer': Custom2DLayer}, {'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
                                           'CustomDenseLayer': CustomDenseLayer}):
    # Use `quantize_apply` to actually make the model quantization aware.
    quant_aware_model = tfmot.quantization.keras.quantize_apply(q_aware_model)

quant_aware_model.summary()

lq.models.summary(quant_aware_model)

print("Quantization Aware Training Starts ===>")

quant_aware_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
print("QAT Model summary ===>")
quant_aware_model.fit(train_images, train_labels, batch_size=64, epochs=6)

_, test_q_aware_model_accuracy = quant_aware_model.evaluate(test_images, test_labels)
print(f"QAT model Test accuracy {test_q_aware_model_accuracy * 100:.2f} %")

print("[INFO] serializing model...")
quant_aware_model.save(args["model"])

print("[INFO] predicting...")
start = time.time()
input_data = test_images[0]
label = test_labels[0]
print("shape of input_data: {}", input_data.shape)
print("input label {}", label)
input_data = np.expand_dims(input_data, axis=0)
print("after dim change shape of input_data: {}", input_data.shape)

preds = quant_aware_model.predict(input_data, batch_size=1).argmax(axis=1)
end = time.time()
print("Elapsed time = {} ms", (end - start) * 1000)
print("preds.shape: {}", preds.shape)
print("preds: {}", preds)
# loop over the sample images

converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()


def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for i, test_image in enumerate(test_images):
        if i % 1000 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy


interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TF test accuracy:', test_q_aware_model_accuracy)

tflite_models_dir = pathlib.Path("tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir / "qat_dnn_8bit.tflite"
tflite_model_quant_file.write_bytes(quantized_tflite_model)

# Measure sizes of models.
_, quant_file = tempfile.mkstemp('.tflite')

with open(quant_file, 'wb') as f:
    f.write(quantized_tflite_model)

# print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2 ** 20))
