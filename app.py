
from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)


model = tf.keras.models.load_model("crop_disease_model.keras")


class_names = [
    "Tomato___Healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        # Load and preprocess image
        image = Image.open(file).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        prediction = class_names[np.argmax(preds)]
        confidence = round(float(np.max(preds) * 100), 2)

    # Load HTML file
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    return render_template_string(
        html,
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
