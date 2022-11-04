import tensorflow as tf

class MyNet(tf.keras.Model):
    def __init__(self, classes = 10):
        super(MyNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(scale=False)
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2))

        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(scale=False)
        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2))

        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(scale=False)
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(64, use_bias=False)
        self.bn4 = tf.keras.layers.BatchNormalization(scale=False)
        self.dense2 = tf.keras.layers.Dense(classes, use_bias=False)
        self.bn5 = tf.keras.layers.BatchNormalization(scale=False)
        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flatten(x)

        x = self.dense1(x)
        x = self.bn4(x)
        x = self.dense2(x)
        x = self.bn5(x)
        x = self.softmax(x)

        return x

