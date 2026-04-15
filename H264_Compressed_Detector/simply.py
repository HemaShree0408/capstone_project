import numpy as np
import cv2
from tensorflow.keras.models import load_model

# --- PATHS ---
MODEL_PATH = r"C:\Users\PESU-RF\capstone 211\capstone_project\H264_Compressed_Detector\best_model.h5"
NPY_PATH   = r"C:\Users\PESU-RF\capstone 211\capstone_project\Human Activity Recognition - Video Dataset__DCT\Meet and Split\Meet and Split (1)\0.npy"

# --- LOAD MODEL ---
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")

# --- LOAD INPUT ---
input_data = np.load(NPY_PATH)

# Ensure input has 3 channels
if input_data.shape[-1] != 3:
    raise ValueError(f"Input must have 3 channels, found {input_data.shape[-1]}")

# Resize to 300x300 for the model
input_resized = cv2.resize(input_data, (300, 300))  # Resize HxW to 300x300
input_resized = np.expand_dims(input_resized, axis=0)  # Add batch dimension
input_resized = input_resized / 255.0  # Normalize if model trained on [0,1]

print("Input shape after preprocessing:", input_resized.shape)

# --- PREDICT ---
pred = model.predict(input_resized)
print("✅ Prediction done!")

# --- RAW OUTPUT ---
print("Raw Prediction Shape:", pred.shape)
print(pred)  # 7x7x5 grid