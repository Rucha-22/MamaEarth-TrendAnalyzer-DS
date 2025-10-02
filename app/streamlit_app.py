import streamlit as st
import pickle
import joblib
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------
# Safe Loader for Pickle/Joblib
# -----------------------------
def load_pipeline(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    # Peek at the first byte
    with open(file_path, "rb") as f:
        first_byte = f.read(1)
    
    try:
        if first_byte == b'\x80':  # likely pickle
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            st.success(f"Loaded '{file_path}' using pickle.")
        else:  # try joblib
            data = joblib.load(file_path)
            st.success(f"Loaded '{file_path}' using joblib.")
        return data
    except Exception as e:
        st.error(f"Failed to load '{file_path}': {e}")
        return None

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model_file = "src/sentiment_model.pkl"       # Replace with your model file path
vectorizer_file = "src/tfidf_vectorizer.pkl" # Your TfidfVectorizer file

model_data = load_pipeline(model_file)
vectorizer = load_pipeline(vectorizer_file)

if model_data is None or vectorizer is None:
    st.stop()

model = model_data['model'] if isinstance(model_data, dict) and 'model' in model_data else model_data

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Mama Earth Reviews Sentiment Analysis")
st.write("Analyze sentiment of customer reviews and visualize insights.")

# Input Section
user_input = st.text_area("Enter a review:", "")

if st.button("Predict Sentiment"):

    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        # Transform text
        X_input = vectorizer.transform([user_input])
        X_input_dense = X_input.toarray()  # Convert sparse to dense for SHAP

        # Predict
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        st.success(f"Predicted Sentiment: {sentiment}")
        st.info(f"Probability: Positive: {prediction_proba[1]:.2f}, Negative: {prediction_proba[0]:.2f}")

        # -----------------------------
        # SHAP Explainability
        # -----------------------------
        st.subheader("SHAP Explanation")
        try:
            explainer = shap.Explainer(model.predict_proba, X_input_dense)
            shap_values = explainer(X_input_dense)

            # SHAP text plot
            st.pyplot(shap.plots.text(shap_values[0], display=False))
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
