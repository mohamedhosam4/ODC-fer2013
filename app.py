import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# Download and load the model
@st.cache_resource  # Cache the model to avoid reloading it multiple times
def download_and_load_model():
    url = 'https://drive.google.com/uc?id=14GDGFpvwkGkh0QG6VIUGGv4BicEDK3x8'  # Replace 'your_file_id_here' with the actual file ID
    output = 'fer2013_model.h5'
    gdown.download(url, output, quiet=False)  # Download the model
    model = tf.keras.models.load_model(output)  # Load the model
    return model

model = download_and_load_model()

# Define emotion labels corresponding to the FER-2013 dataset
emotion_labels = [
    'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'
]

# Streamlit app interface
st.title("Emotion Recognition from Face")
st.write("Upload a face image to predict the emotion.")

# Upload the image
uploaded_file = st.file_uploader("Upload your face image here", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Button to trigger prediction
    if st.button("Predict Emotion"):
        # Preprocess the image
        img = Image.open(uploaded_file).convert("L")  # Convert image to grayscale
        img = img.resize((48, 48))  # Resize the image to 48x48 pixels
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = img_array.reshape(1, 48, 48, 1)  # Reshape the array to match model input

        # Make a prediction using the loaded model
        predictions = model.predict(img_array)
        predicted_emotion = np.argmax(predictions)  # Get the predicted class index

        # Display the result
        st.write(f"Predicted Emotion: **{emotion_labels[predicted_emotion]}**")
else:
    st.write("Please upload a face image.")

# Footer with author information
st.markdown(
    """
    <div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); font-size: 14px; color: gray;">
        This app was created by <strong>Mohamed Hosam</strong>
    </div>
    """, unsafe_allow_html=True
)