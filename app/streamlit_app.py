import streamlit as st
import joblib
import os

# -----------------------------
# Load model safely
# -----------------------------
def load_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None, None
    try:
        saved = joblib.load(file_path)
        st.success(f"Loaded '{file_path}' with joblib.")
        # Extract model and vectorizer
        return saved['model'], saved['vectorizer']
    except Exception as e:
        st.error(f"Failed to load '{file_path}': {e}")
        return None, None

# -----------------------------
# Load pipeline model
# -----------------------------
model_file = "src/best_sentiment_model.pkl"  # Replace with your model path
model, vectorizer = load_model(model_file)
if model is None or vectorizer is None:
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Mama Earth Reviews Sentiment Analysis")
user_input = st.text_area("Enter a review:")

if st.button("Predict Sentiment"):

    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        # Transform input using the saved vectorizer
        X_input = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(X_input)[0]

        # Map numeric prediction to sentiment
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        st.success(f"Sentiment: {sentiment}")
