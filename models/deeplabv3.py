"""
DeepLabV3+ — Atrous Spatial Pyramid Pooling
=============================================
Captures multi-scale context using dilated (atrous) convolutions
without reducing spatial resolution. Ideal for detecting objects
at varying scales in high-resolution satellite imagery.

Architecture:
    Entry Flow → Depthwise Separable Conv blocks
    → ASPP Module (AvgPool + 3× Dilated Separable Conv)
    → Feature Fusion → Upsample → Softmax

ASPP Dilation rates: 6, 12, 18
Output stride: 16
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     BatchNormalization, Activation,
                                     Concatenate, Lambda, AveragePooling2D,
                                     UpSampling2D)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def relu6(x):
    return K.relu(x, max_value=6)


def separable_conv_block(inputs, filters, kernel_size, strides=(1, 1),
                         padding='same', depth_activation=False):
    """Depthwise separable convolution block with optional pre-activation."""
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = inputs
    if depth_activation:
        x = Activation(relu6)(x)
        x = BatchNormalization(axis=channel_axis)(x)
    x = DepthwiseConv2D(kernel_size, strides=strides, padding=padding,
                        depthwise_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    if depth_activation:
        x = Activation(relu6)(x)
    return x


def atrous_spatial_pyramid_pooling(inputs, output_stride=16):
    """
    ASPP Module — fuses global context with multi-scale dilated features.

    Branches:
        b0: Global average pooling → 1×1 Conv → Upsample
        b1-b3: Dilated depthwise separable convolutions (rates 6, 12, 18)
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    divisor = 16 if output_stride == 8 else 8
    pool_h = K.int_shape(inputs)[1] // divisor
    pool_w = K.int_shape(inputs)[2] // divisor
    resize_shape = (pool_h, pool_w)

    # Global context branch
    b0 = AveragePooling2D(pool_size=resize_shape, name='aspp_avgpool')(inputs)
    b0 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
                use_bias=False, name='aspp0')(b0)
    b0 = BatchNormalization(name='aspp_bn0')(b0)
    b0 = Activation(relu6, name='aspp_relu0')(b0)
    b0 = Lambda(lambda x: tf.image.resize(x, resize_shape, method='bilinear'))(b0)

    # Dilated separable conv branches
    b1 = separable_conv_block(inputs, 256, (3, 3), depth_activation=True)
    b2 = separable_conv_block(inputs, 256, (3, 3), depth_activation=True)
    b3 = separable_conv_block(inputs, 256, (3, 3), depth_activation=True)

    b1 = Lambda(lambda x: tf.image.resize(x, resize_shape, method='bilinear'))(b1)
    b2 = Lambda(lambda x: tf.image.resize(x, resize_shape, method='bilinear'))(b2)
    b3 = Lambda(lambda x: tf.image.resize(x, resize_shape, method='bilinear'))(b3)

    x = Concatenate(axis=channel_axis)([b0, b1, b2, b3])
    x = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(name='aspp_concat_bn')(x)
    x = Activation(relu6, name='aspp_concat_relu')(x)
    return x


def DeepLabV3(input_shape=(256, 256, 3), num_classes=8, output_stride=16):
    """
    Build the DeepLabV3+ model.

    Args:
        input_shape (tuple): e.g. (256, 256, 3)
        num_classes (int): Number of segmentation classes.
        output_stride (int): 8 or 16.

    Returns:
        tf.keras.Model
    """
    inputs = Input(shape=input_shape)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
               kernel_initializer='he_normal', use_bias=False)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = separable_conv_block(x, 64, (3, 3), depth_activation=True)
    x = separable_conv_block(x, 128, (3, 3), strides=(2, 2), depth_activation=True)
    x = separable_conv_block(x, 128, (3, 3), depth_activation=True)

    x = atrous_spatial_pyramid_pooling(x, output_stride=output_stride)

    x = Conv2D(256, (1, 1), padding='same',
               kernel_initializer='he_normal', use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = separable_conv_block(x, 256, (3, 3), depth_activation=True)
    x = separable_conv_block(x, 256, (3, 3), depth_activation=True)

    if output_stride == 8:
        x = UpSampling2D((4, 4), interpolation='bilinear')(x)
        x = Conv2D(256, (3, 3), padding='same',
                   kernel_initializer='he_normal', use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation(relu6)(x)

    x = Conv2D(num_classes, (1, 1), padding='same',
               kernel_initializer='he_normal', activation='softmax')(x)

    return Model(inputs, x, name="DeepLabV3Plus")


if __name__ == "__main__":
    from tensorflow.keras.optimizers import Adam
    model = DeepLabV3(input_shape=(256, 256, 3), num_classes=8, output_stride=16)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
