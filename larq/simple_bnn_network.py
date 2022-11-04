import tensorflow as tf
import larq as lq

# All quantized layers except the first will use the same options
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")

class MyBnnNet(tf.keras.Model):
    def __init__(self, classes=10):
        super(MyBnnNet, self).__init__()
        self.conv1 = lq.layers.QuantConv2D(32, (3, 3),
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                use_bias=False)
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False)

        self.conv2 = lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs)
        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False)

        self.conv3 = lq.layers.QuantConv2D(64, (3, 3), use_bias=False, **kwargs)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = lq.layers.QuantDense(64, use_bias=False, **kwargs)
        self.bn4 = tf.keras.layers.BatchNormalization(scale=False)

        self.dense2 = lq.layers.QuantDense(classes, use_bias=False, **kwargs)
        self.bn5 = tf.keras.layers.BatchNormalization(scale=False)
        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn4(x)

        x = self.dense2(x)
        x = self.bn5(x)
        x = self.softmax(x)

        return x
