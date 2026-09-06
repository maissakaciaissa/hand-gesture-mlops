import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋")

st.title("✋ Hand Gesture Recognition")
st.write("Upload a hand gesture image and see what the model predicts.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            image_bytes = uploaded_file.getvalue()
            files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}

            try:
                response = requests.post(API_URL, files=files, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Prediction:** {result['label']}")
                    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
                elif response.status_code == 429:
                    st.error("Rate limit exceeded — please wait a moment and try again.")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is it running?")