
import streamlit as st
import joblib
import pandas as pd

# Load the trained Naive Bayes model and TF-IDF vectorizer
try:
    loaded_model = joblib.load('best_naive_bayes_sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    st.success("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Make sure 'best_naive_bayes_sentiment_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    loaded_model = None
    vectorizer = None

# Define the Streamlit app
st.title("VR Motion Sickness Level Predictor")
st.header("Predict the Motion Sicknes of a VR .")

if loaded_model and vectorizer:
    # Create a text area for user input
    review_text = st.text_area("Enter data here:", "")

    # Add a button to trigger the prediction
    if st.button("Predict Sentiment"):
        if review_text:
            # Preprocess the input text
            # The vectorizer expects an iterable (like a list)
            review_text_tfidf = vectorizer.transform([review_text])

            # Make a prediction
            prediction = loaded_model.predict(review_text_tfidf)

            # Display the result
            sentiment_label = "Positive Immersion" if prediction[0] == 1 else "Negative Immersion"

            st.subheader("Prediction:")
            if prediction[0] == 1:
                st.success(f"Predicted Sentiment: {sentiment_label}")
            else:
                st.error(f"Predicted Sentiment: {sentiment_label}")
        else:
            st.warning("Please enter data to predict .")

st.info("This app predicts whether a VR Motion Sickness base on age and other data.The Model trained for better experience play with VR.")
