
import streamlit as st
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

# 1. Streamlit page configuration
st.set_page_config(
    page_title="X-Ray Image Classifier",
    page_icon="🩻",
    layout="centered"
)
# 2. Application header
html_temp = """
<div style="background-color:red; padding:10px; border-radius:8px;">
    <h2 style="color:white; text-align:center;">
        X-Ray Image Classifier
    </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
st.write("")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

# 3. Model configuration
img_size = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

MODEL_PATH = (r"C:\Users\HP\Desktop\New ML\X-ray-detection\pretrained_model.h5")
# 4. Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()
# 5. Upload and classify an X-ray image
def load_classifier():
    st.subheader("Upload an X-ray Image")
    file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:

        try:
            # Read image and convert to RGB (3 channels for VGG16)
            img = Image.open(BytesIO(file.getvalue())).convert("RGB")

            # Resize to the model's expected dimensions
            img = img.resize((img_size, img_size))

            # Display uploaded image
            st.image(
                img,
                caption="Uploaded X-ray Image",
                width=300
            )

            # Convert image to NumPy array
            image_array = tf.keras.preprocessing.image.img_to_array(img)

            # Add batch dimension: (100, 100, 3) -> (1, 100, 100, 3)
            image_array = np.expand_dims(image_array, axis=0)

            # Normalize pixel values to 0-1
            image_array = image_array / 255.0

            # Prediction button
            if st.button("PREDICT", type="primary"):

                with st.spinner("Analyzing image..."):

                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )

                    # Assumes sigmoid output:
                    # 0 = NORMAL, 1 = PNEUMONIA
                    probability = float(prediction[0][0])

                    predicted_class = (
                        1 if probability >= 0.5 else 0
                    )

                    confidence = (
                        probability
                        if predicted_class == 1
                        else 1 - probability
                    ) * 100

                # Display result
                st.subheader("Prediction Result")

                if predicted_class == 1:
                    st.error("Prediction: PNEUMONIA")
                else:
                    st.success("Prediction: NORMAL")

                st.write(f"Model score for pneumonia: {probability:.2%}")
                st.write(f"Confidence in predicted class: {confidence:.2f}%")

                st.caption(
                    "Educational demonstration only. This model is not "
                    "a substitute for professional medical diagnosis."
                )

        except Exception as e:
            st.error(f"Error processing the image: {e}")

# 6. Run application
def main():
    load_classifier()
if __name__ == "__main__":
    main()

