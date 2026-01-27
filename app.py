import streamlit as st
import numpy as np
from PIL import Image
from model import load_trained_model

# ---------------- CONFIG ----------------
CONFIDENCE_THRESHOLD = 80  # %
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Healthy", "Diseased"]

# ---------------- UI ----------------
st.set_page_config(page_title="AI Crop Health System", layout="centered")
st.title("🌱 AI Crop Health Decision Support System")
st.write("Upload a **tomato leaf image** to detect crop health status.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return load_trained_model()

model = load_model()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload crop leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        # ---------- IMAGE PROCESS ----------
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ---------- PREDICTION ----------
        prediction = model.predict(img_array)
        confidence = np.max(prediction) * 100
        class_id = np.argmax(prediction)

        # ---------- DECISION LOGIC ----------
        st.subheader("🔍 Prediction Result")

        if confidence < CONFIDENCE_THRESHOLD:
            st.error(
                "⚠️ **Unsupported or unclear image detected**\n\n"
                "This does NOT look like a clear tomato leaf image.\n"
                "Please upload a proper tomato leaf photo."
            )
            st.write(f"Model confidence too low: **{confidence:.2f}%**")

        else:
            st.write(f"📊 Confidence: **{confidence:.2f}%**")

            if CLASS_NAMES[class_id] == "Diseased":
                st.error("🦠 **Prediction: Diseased**")
                st.warning("💊 Recommended Action: Apply appropriate pesticide or consult an agronomist.")
            else:
                st.success("✅ **Prediction: Healthy**")
                st.success("🌿 Crop is healthy. No action needed.")

    except Exception as e:
        st.error("❌ Error processing the image.")
        st.write(str(e))
