"""
U-Net — Encoder-Decoder with Symmetric Skip Connections
=========================================================
Classic architecture for semantic segmentation.
Used as the baseline model for comparison against ResUNet,
DeepLabV3+, and PSPNet in satellite scene segmentation.

Architecture:
    Encoder: [Conv×2 → Pool] × 4  (64→128→256→512→1024 filters)
    Decoder: [TransposeConv + SkipConcat + Conv×2] × 4
    Output:  Softmax (num_classes)

Reference:
    Ronneberger et al. (2015) "U-Net: Convolutional Networks for
    Biomedical Image Segmentation"
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
                                     Concatenate, Conv2DTranspose)
from tensorflow.keras.optimizers import Adam


class ImageSegmentationModels:
    """
    Container for U-Net and CNN segmentation models with shared data loading.

    Args:
        input_shape (tuple): e.g. (256, 256, 3)
        image_folder (str): Path to image tiles.
        mask_folder (str): Path to segmentation masks.
    """

    def __init__(self, input_shape, image_folder, mask_folder):
        self.input_shape = input_shape
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.cnn_model = self.create_cnn_model()
        self.unet_model = self.create_unet_model()

    def create_cnn_model(self):
        """
        Build a simple CNN encoder (feature extractor, no decoder).
        Used for classification or as a backbone.
        """
        inputs = Input(self.input_shape)
        x = Conv2D(32,  (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64,  (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        return Model(inputs=inputs, outputs=x, name="CNN_Encoder")

    def create_unet_model(self):
        """
        Build the full U-Net segmentation model.

        Returns:
            tf.keras.Model: U-Net with sigmoid output for binary segmentation.
                            For multi-class, replace sigmoid with softmax and
                            adjust output channels to num_classes.
        """
        inputs = Input(self.input_shape)

        # Encoder
        c1 = Conv2D(64,   (3, 3), activation='relu', padding='same')(inputs)
        c1 = Conv2D(64,   (3, 3), activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(128,  (3, 3), activation='relu', padding='same')(p1)
        c2 = Conv2D(128,  (3, 3), activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(256,  (3, 3), activation='relu', padding='same')(p2)
        c3 = Conv2D(256,  (3, 3), activation='relu', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(512,  (3, 3), activation='relu', padding='same')(p3)
        c4 = Conv2D(512,  (3, 3), activation='relu', padding='same')(c4)
        p4 = MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
        c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

        # Decoder
        u6 = Concatenate(axis=3)([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5), c4])
        c6 = Conv2D(512,  (3, 3), activation='relu', padding='same')(u6)
        c6 = Conv2D(512,  (3, 3), activation='relu', padding='same')(c6)

        u7 = Concatenate(axis=3)([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6), c3])
        c7 = Conv2D(256,  (3, 3), activation='relu', padding='same')(u7)
        c7 = Conv2D(256,  (3, 3), activation='relu', padding='same')(c7)

        u8 = Concatenate(axis=3)([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7), c2])
        c8 = Conv2D(128,  (3, 3), activation='relu', padding='same')(u8)
        c8 = Conv2D(128,  (3, 3), activation='relu', padding='same')(c8)

        u9 = Concatenate(axis=3)([Conv2DTranspose(64,  (2, 2), strides=(2, 2), padding='same')(c8), c1])
        c9 = Conv2D(64,   (3, 3), activation='relu', padding='same')(u9)
        c9 = Conv2D(64,   (3, 3), activation='relu', padding='same')(c9)

        # Output — sigmoid for binary, softmax for multi-class
        outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)

        return Model(inputs=inputs, outputs=outputs, name="UNet")

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        """Compile and train both CNN and U-Net models."""
        for name, m in [("CNN", self.cnn_model), ("UNet", self.unet_model)]:
            print(f"\nTraining {name}...")
            m.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            m.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_val, y_val))

    def evaluate(self, X_test, y_test):
        """Evaluate both models on the test set."""
        for name, m in [("CNN", self.cnn_model), ("UNet", self.unet_model)]:
            loss, acc = m.evaluate(X_test, y_test, verbose=0)
            print(f"{name} — Loss: {loss:.4f} | Accuracy: {acc:.4f}")


if __name__ == "__main__":
    seg = ImageSegmentationModels(
        input_shape=(256, 256, 3),
        image_folder="./data/train/images",
        mask_folder="./data/train/labels"
    )
    seg.unet_model.summary()
