import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

model_name = "imgmodel.h5"

if os.path.exists(model_name):
    model = load_model(model_name)
else:
    st.error("model not found.")
    st.stop()

st.title("Cat,Dog Classifier")
st.write("Upload an image:")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = image.load_img(file, target_size=(128, 128))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.success("Prediction: 🐶 Dog")
    else:
        st.success("Prediction: 🐱 Cat")
