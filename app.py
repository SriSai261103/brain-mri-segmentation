import streamlit as st
from PIL import Image
import requests

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI.", use_column_width=True)
    
    # Send the image to FastAPI for segmentation
    response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file})
    
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.image(prediction, caption="Segmentation Result", use_column_width=True)
