from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from Resnet18 import Resnet18
from PSPNet import PSPNet
import tensorflow as tf

class VRM(Model):
    def __init__(self):
        super().__init__()

        self.model = Sequential()
        self.model.add(Dense(625))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dense(625))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())

    def call(self, inputs, training=None, mask=None):
        output = self.model(inputs)

        return output


class EncoderWithVTM(Model):
    def __init__(self, V):
        super().__init__()
        self.V = V

        # Build encoder network
        self.resnet = Resnet18()
        self.max_pool = MaxPooling2D(pool_size=(8, 8))

        # Build VRM
        self.vrm_list = []
        for i in range(self.V):
            self.vrm_list.append(VRM())

    def call(self, inputs, training=None, mask=None):
        # input shape (B ,V, H, W, C)
        B, V, H, W, C = inputs.get_shape()

        # Reshape inputs (B, N, H, W, C)
        output = tf.reshape(inputs, shape=(B * V, H, W, C))

        # Resnet18 Output shape (B * V, 200, 200, 512)
        output = self.resnet(output)

        # Maxpooling Output shape (B * V, 25, 25, 512)
        output = self.max_pool(output)

        # Transpose and reshape (B, V, H, W, C) to (B, V, C, H * W)
        # Output shape (B, V, 512, 625)
        B_V, H, W, C = output.shape
        output = tf.reshape(output, (B, V, H, W, C))
        output = tf.transpose(output, (0, 1, 4, 2, 3))
        output = tf.reshape(output, (B, V, C, H * W))

        # VRM Output shape (1, 512, 625)
        vrm_output = []
        for i in range(self.V):
            feature = tf.expand_dims(output[:, i, :, :], 0)
            vrm_output.append(self.vrm_list[i](feature))

        # Fusion Output shape (1, 512, 625)
        output = tf.zeros(shape=(1, 128, 625))
        for i in range(self.V):
            output += vrm_output[i]

        # Reshape (1, 512, 625) to (1, 25, 25, 512)
        output = tf.reshape(output, (B, C, H, W))
        output = tf.transpose(output, (0, 2, 3, 1))

        return output


class VPN(Model):
    def __init__(self, num_class, V, M):
        super().__init__()
        """
        :param num_class: Class of need predict
        :param V: Number of View
        :param M: Number of Method
        """
        self.num_class = num_class
        self.V = V
        self.M = M

        # Build Encoder with VTM
        self.encoder_with_vtm = []
        for i in range(self.M):
            self.encoder_with_vtm.append(EncoderWithVTM(self.V))

        # PSPNet
        self.psp_net = PSPNet(num_class=num_class)

    def call(self, inputs, training=None, mask=None):
        vtm_output = tf.zeros(shape=(1, 25, 25, 128))

        for i in range(self.M):
            vtm_output += self.encoder_with_vtm[i](inputs[:, i * self.V: (i + 1) * self.V, :, :, :])

        # PSPNet output shape (1, 200, 200, num_cls)
        output = self.psp_net(vtm_output)

        return output
