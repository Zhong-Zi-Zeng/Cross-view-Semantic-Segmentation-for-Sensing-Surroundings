from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf


class ResnetBlock(Model):
    def __init__(self, filters, strides=1, same_dim=False):
        super().__init__()
        self.filters = filters
        self.strides = strides
        # 是否與輸入層相同維度
        self.same_dim = same_dim

        self.c1 = Conv2D(self.filters, (3, 3), strides=self.strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(self.filters, (3, 3), strides=self.strides, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        if not self.same_dim:
            self.down_c1 = Conv2D(self.filters, (1, 1), strides=self.strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs

        # 第一個區塊
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        # 第二個區塊
        x = self.c2(x)
        y = self.b2(x)

        if not self.same_dim:
            residual = self.down_c1(inputs)
            residual = self.b1(residual)

        out = self.a2(y + residual)

        return out

class Resnet18(Model):
    def __init__(self):
        super().__init__()
        self.layerSequential = Sequential()
        self.filters_list = [64, 128]

        self.c1 = Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        for block_id in range(len(self.filters_list)):
            for i in range(2):
                if block_id != 0 and i == 0:
                    layer = ResnetBlock(filters=self.filters_list[block_id], same_dim=False)
                else:
                    layer = ResnetBlock(filters=self.filters_list[block_id], same_dim=True)
                self.layerSequential.add(layer)

    def call(self, input):
        output = self.c1(input)
        output = self.b1(output)
        output = self.a1(output)
        output = self.layerSequential(output)  # (H, W, 256)

        return output



