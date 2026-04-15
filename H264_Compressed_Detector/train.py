import os
import numpy as np
import tensorflow as tf
from model import SSD300_H264
from data_loader import H264DatasetGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- CONFIG ---
DATASET_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\FINAL_SSD_DATASET"
BATCH_SIZE = 32
EPOCHS = 50

def train():
    # --- GPU CHECK ---
    print("\n" + "="*40)
    print("🛠️ Checking System Resources...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Training on GPU: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("="*40 + "\n")

    # --- 1. MODEL ---
    model = SSD300_H264(n_classes=1, image_shape=(300, 300, 3))

    # --- 2. LOSS FUNCTION (CUSTOM YOLO STYLE) ---
    def yolo_loss(y_true, y_pred):
        # y_true, y_pred shape: (batch, 7,7,5)

        obj_mask = y_true[..., 0]  # confidence

        # --- LOSS PARTS ---
        # Confidence loss
        conf_loss = tf.reduce_sum(tf.square(y_true[..., 0] - y_pred[..., 0]))

        # Box loss (only where object exists)
        box_loss = tf.reduce_sum(
            obj_mask * tf.reduce_sum(tf.square(y_true[..., 1:] - y_pred[..., 1:]), axis=-1)
        )

        return conf_loss + box_loss

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=yolo_loss
    )

    # --- 3. DATA GENERATOR ---
    train_gen = H264DatasetGenerator(DATASET_PATH, mode="train", batch_size=BATCH_SIZE)
    val_gen = H264DatasetGenerator(DATASET_PATH, mode="val", batch_size=BATCH_SIZE)

    # --- 4. CALLBACKS ---
    callbacks = [
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss", verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
    ]

    # --- 5. TRAIN ---
    print("📈 Starting Training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    model.save("final_model.h5")
    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    train()