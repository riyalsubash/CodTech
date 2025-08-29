import streamlit as st
import joblib

model = joblib.load("sentiment.pkl")
vectorizer = joblib.load("vect.pkl")

st.title("Sentiment Analysis App")
st.write("Enter text to analyze sentiment:")

user_input = st.text_area("Type a sentence...")

if st.button("Predict"):
    text_vec = vectorizer.transform([user_input])
    pred = model.predict(text_vec)[0]
    st.write("### Predicted Sentiment:", pred)
