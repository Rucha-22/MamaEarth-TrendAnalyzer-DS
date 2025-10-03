import streamlit as st
import joblib
import shap
import os
import re

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

        # Some models like LinearSVC may not have predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            st.info(f"Probability → Positive: {proba[2]:.2f}, Neutral: {proba[1]:.2f}, Negative: {proba[0]:.2f}")
        else:
            st.info("Probability info not available for this model.")

        # Map numeric prediction to sentiment
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(prediction, "Unknown")
        st.success(f"Sentiment: {sentiment}")

        # -----------------------------
        # Optional SHAP explanation
        # -----------------------------
        st.subheader("Word-Level Sentiment Highlights")
        try:
            explainer = shap.Explainer(model, X_input, algorithm="partition")
            shap_values = explainer(X_input)

            # Map words to SHAP contributions
            word_vals = dict(zip(shap_values[0].data, shap_values[0].values))

            # Highlight words in text
            def highlight_text(text, word_vals):
                def replacer(match):
                    word = match.group(0)
                    value = word_vals.get(word, 0)
                    if value > 0:
                        return f'<span style="background-color: #b6fcb6">{word}</span>'
                    elif value < 0:
                        return f'<span style="background-color: #fcb6b6">{word}</span>'
                    else:
                        return word
                return re.sub(r'\b\w+\b', replacer, text)

            highlighted_review = highlight_text(user_input, word_vals)
            st.markdown(highlighted_review, unsafe_allow_html=True)

            # Optional: SHAP text plot
            st.subheader("SHAP Text Plot")
            st.pyplot(shap.plots.text(shap_values[0], display=False))

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
