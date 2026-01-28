import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("crop_disease_model.keras")

class_names = [
    "Tomato___Healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold"
]

img_path = "dataset/Tomato___Healthy/leaf.jpg"

img = Image.open(img_path).resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print("Prediction:", class_names[np.argmax(pred)])
print("Confidence:", np.max(pred) * 100)
