"""
ResUNet — U-Net with Residual Blocks
=====================================
Encoder-decoder architecture with residual skip connections.
Residual blocks help preserve spatial features across the bottleneck,
critical for fine-grained satellite scene segmentation.

Architecture:
    Encoder: Conv → ResBlock(16) → Pool → ResBlock(32) → Pool
             → ResBlock(64) → Pool → ResBlock(128) → Pool → Conv(256)
    Decoder: TransposeConv + SkipConcat → ×4
    Output:  Softmax (num_classes)

Reference:
    Zhang et al. (2018) "Road Extraction by Deep Residual U-Net"
    Applied here for multi-class satellite image segmentation.
"""

import os
import cv2
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ------------------------------------------------------------------ #
#  Data Loading                                                        #
# ------------------------------------------------------------------ #

def load_images_and_labels(folder_path):
    """
    Load paired satellite images and segmentation label masks.

    Expects:
        folder_path/
            images/   ← RGB satellite image tiles (.png / .jpg)
            labels/   ← Grayscale segmentation masks (pixel = class ID)

    Args:
        folder_path (str): Path to train / test / validation split folder.

    Returns:
        tuple[np.ndarray, np.ndarray]: (images [N,H,W,3], labels [N,H,W])
    """
    image_folder = os.path.join(folder_path, 'images')
    label_folder = os.path.join(folder_path, 'labels')
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))
    images, labels = [], []
    for img_f, lbl_f in zip(image_files, label_files):
        images.append(cv2.imread(os.path.join(image_folder, img_f)))
        labels.append(cv2.imread(os.path.join(label_folder, lbl_f), cv2.IMREAD_GRAYSCALE))
    return np.array(images), np.array(labels)


# ------------------------------------------------------------------ #
#  Model Blocks                                                        #
# ------------------------------------------------------------------ #

def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = layers.Activation('relu')(x)
    return x


def residual_block(block_input, num_filters=16):
    x = layers.Activation('relu')(block_input)
    x = layers.BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Concatenate()([x, block_input])
    return x


# ------------------------------------------------------------------ #
#  ResUNet                                                             #
# ------------------------------------------------------------------ #

def ResUNet(input_shape, num_classes):
    """
    Build the ResUNet model.

    Args:
        input_shape (tuple): e.g. (256, 256, 3)
        num_classes (int): Number of segmentation classes.

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = convolution_block(inputs, 16, (3, 3))
    r1 = residual_block(c1, 16)
    p1 = layers.MaxPooling2D((2, 2))(r1)

    c2 = convolution_block(p1, 32, (3, 3))
    r2 = residual_block(c2, 32)
    p2 = layers.MaxPooling2D((2, 2))(r2)

    c3 = convolution_block(p2, 64, (3, 3))
    r3 = residual_block(c3, 64)
    p3 = layers.MaxPooling2D((2, 2))(r3)

    c4 = convolution_block(p3, 128, (3, 3))
    r4 = residual_block(c4, 128)
    p4 = layers.MaxPooling2D((2, 2))(r4)

    c5 = convolution_block(p4, 256, (3, 3))

    # Decoder
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, r4])
    c6 = convolution_block(u6, 128, (3, 3))

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, r3])
    c7 = convolution_block(u7, 64, (3, 3))

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, r2])
    c8 = convolution_block(u8, 32, (3, 3))

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, r1], axis=3)
    c9 = convolution_block(u9, 16, (3, 3))

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    return Model(inputs=[inputs], outputs=[outputs], name="ResUNet")


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    DATA_ROOT = './data'
    NUM_CLASSES = 8
    INPUT_SHAPE = (256, 256, 3)
    BATCH_SIZE = 8
    EPOCHS = 50

    train_images, train_labels = load_images_and_labels(f'{DATA_ROOT}/train')
    val_images, val_labels = load_images_and_labels(f'{DATA_ROOT}/validation')

    train_labels_oh = to_categorical(train_labels, num_classes=NUM_CLASSES)
    val_labels_oh = to_categorical(val_labels, num_classes=NUM_CLASSES)

    train_gen = ImageDataGenerator(rescale=1./255).flow(train_images, train_labels_oh, batch_size=BATCH_SIZE)
    val_gen = ImageDataGenerator(rescale=1./255).flow(val_images, val_labels_oh, batch_size=BATCH_SIZE)

    model = ResUNet(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint("resunet_best.keras", monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_images) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_images) // BATCH_SIZE,
        callbacks=callbacks
    )
