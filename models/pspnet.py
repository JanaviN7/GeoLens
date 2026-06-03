"""
PSPNet — Pyramid Scene Parsing Network
========================================
Aggregates global and local context at multiple scales using a
Pyramid Pooling Module (PPM). Two variants are provided:

  PSPNet_v1: Standard PPM with pool sizes [1, 2, 3, 6]
  PSPNet_v2: Extended PPM with dilated convolutions (rates 6, 12, 18)
             for richer multi-scale feature extraction.

Both trained for 8-class satellite semantic segmentation at 256×256.

Reference:
    Zhao et al. (2017) "Pyramid Scene Parsing Network" (CVPR)
"""

from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Concatenate, UpSampling2D,
                                     GlobalAveragePooling2D, Reshape,
                                     BatchNormalization, Activation,
                                     AveragePooling2D)
from tensorflow.keras.models import Model


# ------------------------------------------------------------------ #
#  Shared helpers                                                      #
# ------------------------------------------------------------------ #

def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', padding='same'):
    x = Conv2D(filters, kernel_size, activation=None, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x


def _encoder(inputs):
    """Shared 5-stage encoder for both PSPNet variants."""
    c1 = conv_block(inputs, 64);   p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 128);      p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 256);      p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 512);      p4 = MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 1024)
    return c1, c2, c3, c4, c5


# ------------------------------------------------------------------ #
#  PSPNet v1 — Standard Pyramid Pooling                               #
# ------------------------------------------------------------------ #

def psp_module_v1(inputs, out_channels):
    """
    Standard Pyramid Pooling Module.

    Fuses:
      - Global average pool (full scene context)
      - AveragePool at scales 1, 2, 3, 6 → 1×1 Conv → Upsample
    """
    branches = [GlobalAveragePooling2D()(inputs)]
    for pool_size in [1, 2, 3, 6]:
        b = AveragePooling2D(pool_size=(pool_size, pool_size))(inputs)
        b = conv_block(b, out_channels, kernel_size=(1, 1))
        b = UpSampling2D(size=(pool_size, pool_size), interpolation='bilinear')(b)
        branches.append(b)
    return Concatenate()(branches)


def build_pspnet_v1(input_shape=(256, 256, 3), num_classes=8):
    """
    PSPNet v1 — encoder + standard PPM + decoder.

    Args:
        input_shape (tuple): e.g. (256, 256, 3)
        num_classes (int): Segmentation classes.

    Returns:
        tf.keras.Model
    """
    inputs = Input(shape=input_shape)
    c1, c2, c3, c4, c5 = _encoder(inputs)

    psp = psp_module_v1(c5, 256)

    up1 = UpSampling2D(size=(8, 8), interpolation='bilinear')(psp)
    x = conv_block(up1, 256)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    return Model(inputs, outputs, name="PSPNet_v1")


# ------------------------------------------------------------------ #
#  PSPNet v2 — Dilated Convolution PPM                                #
# ------------------------------------------------------------------ #

def psp_module_v2(inputs, out_channels):
    """
    Extended PPM with dilated convolutions.

    Branches:
      b1: Global AvgPool → reshape → 1×1 Conv → Upsample
      b2: 1×1 Conv (local)
      b3-b5: Dilated 3×3 Conv (rates 6, 12, 18)
    """
    h = inputs.shape[1]
    w = inputs.shape[2]

    b1 = GlobalAveragePooling2D()(inputs)
    b1 = Reshape((1, 1, inputs.shape[-1]))(b1)
    b1 = Conv2D(out_channels, (1, 1), activation='relu')(b1)
    b1 = UpSampling2D(size=(h, w))(b1)

    b2 = Conv2D(out_channels, (1, 1), activation='relu')(inputs)
    b3 = Conv2D(out_channels, (3, 3), activation='relu', padding='same', dilation_rate=6)(inputs)
    b4 = Conv2D(out_channels, (3, 3), activation='relu', padding='same', dilation_rate=12)(inputs)
    b5 = Conv2D(out_channels, (3, 3), activation='relu', padding='same', dilation_rate=18)(inputs)

    return Concatenate(axis=-1)([b1, b2, b3, b4, b5])


def build_pspnet_v2(input_shape=(256, 256, 3), num_classes=8):
    """
    PSPNet v2 — encoder + dilated PPM + skip-connection decoder.

    Args:
        input_shape (tuple): e.g. (256, 256, 3)
        num_classes (int): Segmentation classes.

    Returns:
        tf.keras.Model
    """
    inputs = Input(input_shape)
    c1, c2, c3, c4, c5 = _encoder(inputs)

    psp = psp_module_v2(c5, 256)

    up6 = Conv2D(512, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(psp))
    m6 = Concatenate(axis=-1)([c4, up6])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(m6)

    up7 = Conv2D(256, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(c6))
    m7 = Concatenate(axis=-1)([c3, up7])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(m7)

    up8 = Conv2D(128, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(c7))
    m8 = Concatenate(axis=-1)([c2, up8])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(m8)

    up9 = Conv2D(64, (3, 3), activation='relu', padding='same')(UpSampling2D((2, 2))(c8))
    m9 = Concatenate(axis=-1)([c1, up9])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(m9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    return Model(inputs, outputs, name="PSPNet_v2")


if __name__ == "__main__":
    m1 = build_pspnet_v1(input_shape=(256, 256, 3), num_classes=8)
    m1.summary()
    m2 = build_pspnet_v2(input_shape=(256, 256, 3), num_classes=8)
    m2.summary()
