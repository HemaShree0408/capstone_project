import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

def SSD300_H264(n_classes=1, image_shape=(300, 300, 3)):
    """
    Simplified YOLO-style model for DCT input
    Output: (7, 7, 5)
    Each cell → [objectness, cx, cy, w, h]
    """

    inputs = Input(shape=image_shape)

    # --- Backbone ---
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)

    # --- Detection Head ---
    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(7 * 7 * 5, activation='sigmoid')(x)

    outputs = tf.keras.layers.Reshape((7, 7, 5))(x)

    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = SSD300_H264()
    model.summary()