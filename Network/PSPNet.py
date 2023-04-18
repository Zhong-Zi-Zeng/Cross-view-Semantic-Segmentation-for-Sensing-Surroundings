from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf


class PyramidPool(Model):
    def __init__(self, pool_size):
        super().__init__()

        self.model = Sequential()

        self.model.add(AveragePooling2D(pool_size=pool_size))
        self.model.add(Conv2D(16, kernel_size=(3, 3), padding='same', use_bias=False))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

        self.one_conv = Conv2D(128 // 4, kernel_size=(1, 1), padding='same', use_bias=False)

    def call(self, inputs, training=None, mask=None):
        output = self.model(inputs)
        output = tf.image.resize(output, size=(inputs.shape[1], inputs.shape[2]))
        output = self.one_conv(output)

        return output


class PSPNet(Model):
    def __init__(self, num_class):
        super().__init__()

        self.l1 = PyramidPool(1)
        self.l2 = PyramidPool(2)
        self.l3 = PyramidPool(3)
        self.l4 = PyramidPool(6)

        self.upper_sample = Sequential()
        self.upper_sample.add(UpSampling2D((8, 8)))
        self.upper_sample.add(Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        self.upper_sample.add(BatchNormalization())
        self.upper_sample.add(LeakyReLU())
        self.upper_sample.add(Conv2DTranspose(num_class, (3, 3), strides=(1, 1), padding='same', use_bias=False))

    def call(self, inputs, training=None, mask=None):
        # input shape (1, 25, 25, 512)
        output_1 = self.l1(inputs)  # output shape (1, 25, 25, 128)
        output_2 = self.l2(inputs)  # output shape (1, 25, 25, 128)
        output_3 = self.l3(inputs)  # output shape (1, 25, 25, 128)
        output_4 = self.l4(inputs)  # output shape (1, 25, 25, 128)

        # output shape (1, 25, 25, 1024)
        output = tf.concat([inputs, output_1, output_2, output_3, output_4], 3)

        # output shape (1, 200, 200, num_cls)
        output = self.upper_sample(output)

        return output
