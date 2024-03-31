import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained model
model = load_model('fake_news_detection_model.h5')

# Streamlit App
st.title("Fake News Detection App")
st.write("Enter the news text below to check if it's fake or true.")

# Add text input for user
user_text = st.text_area("Enter news text here:")

# Function to preprocess and predict
def predict_fake_news(text):
    # Preprocess the input text
    tokenizer = Tokenizer()
    tokenizer.word_index = np.load('tokenizer_word_index.npy', allow_pickle=True).item()
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)  # Ensure maxlen is consistent
    # Make prediction
    prediction = model.predict(padded_sequence)
    return prediction[0][0]

# Make prediction when the button is clicked
if st.button("Check"):
    if user_text.strip() == "":
        st.error("Please enter some text.")
    else:
        prediction = predict_fake_news(user_text)
        st.write(f"Prediction value: {prediction}")
        if prediction >= 0.5:
            st.error("Fake News!")
        else:
            st.success("True News!")
