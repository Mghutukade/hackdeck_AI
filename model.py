import os
import tensorflow as tf

MODEL_PATH = "trained_model.h5"

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Trained model not found. Run train_model.py first.")
    return tf.keras.models.load_model(MODEL_PATH)
