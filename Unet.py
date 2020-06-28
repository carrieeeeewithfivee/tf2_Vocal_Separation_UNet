import tensorflow as tf
from config import *
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model

print('TensorFlow Version: ', tf.__version__)

class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides, name=None, padding=1, dilation_rate=1):
        super(Conv2D, self).__init__(name=name)
        
        self.conv_out = layers.Conv2D(filters=filters,
                                      kernel_size=kernel_size, 
                                      strides=strides, 
                                      padding='same',
                                      kernel_initializer='he_normal',
                                      dilation_rate=dilation_rate)
        self.batchNorm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        x = self.conv_out(inputs)
        x = self.batchNorm(x)
        x = self.activation(x)
        return x

class DeConv2D(layers.Layer):
    def __init__(self, filters, dropout = False, batchNorm = True, kernel_size=4, strides=2, name=None):
        super(DeConv2D, self).__init__(name=name)
        self.drop = dropout
        self.Norm = batchNorm
        self.deconv_out = layers.Conv2DTranspose(filters=filters,
                                                kernel_size=kernel_size, 
                                                strides=strides, 
                                                padding='same',
                                                name=name)
        self.batchNorm = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs):
        x = self.deconv_out(inputs)
        if self.Norm == True:
            x = self.batchNorm(x)
        if self.drop == True:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class Unet(tf.keras.Model):
    def __init__(self, max_displacement=4, batchNorm=True):
        super(Unet, self).__init__()
    
        self.conv1 = Conv2D(16, kernel_size=5, strides=2)
        self.conv2 = Conv2D(32, kernel_size=5, strides=2)
        self.conv3 = Conv2D(64, kernel_size=5, strides=2)
        self.conv4 = Conv2D(128, kernel_size=5, strides=2)
        self.conv5 = Conv2D(256, kernel_size=5, strides=2)
        self.conv6 = Conv2D(512, kernel_size=5, strides=2)

        self.deconv1 = DeConv2D(256, kernel_size=5, strides=2, dropout=True) 
        self.deconv2 = DeConv2D(128, kernel_size=5, strides=2, dropout=True) 
        self.deconv3 = DeConv2D(64, kernel_size=5, strides=2, dropout=True) 
        self.deconv4 = DeConv2D(32, kernel_size=5, strides=2) 
        self.deconv5 = DeConv2D(16, kernel_size=5, strides=2) 
        self.deconv6 = DeConv2D(1, kernel_size=5, strides=2, batchNorm=False)

    def call(self, inputs, is_training=True):
        #input size: (512, 128, 1)
        #downsample
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        #upsample
        deconv1 = self.deconv1(conv6)

        deconv2 = tf.concat([deconv1, conv5], 3)
        #predict2 = self.predict2(deconv2)
        deconv2 = self.deconv2(deconv2)

        deconv3 = tf.concat([deconv2, conv4], 3)
        #predict3 = self.predict3(deconv3)
        deconv3 = self.deconv3(deconv3)

        deconv4 = tf.concat([deconv3, conv3], 3)
        #predict4 = self.predict4(deconv4)
        deconv4 = self.deconv4(deconv4)

        deconv5 = tf.concat([deconv4, conv2], 3)
        #predict5 = self.predict5(deconv5)
        deconv5 = self.deconv5(deconv5)

        deconv6 = tf.concat([deconv5, conv1], 3)
        predict6 = self.deconv6(deconv6)

        return predict6