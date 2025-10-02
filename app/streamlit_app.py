import streamlit as st
import pickle
import shap
import matplotlib.pyplot as plt

# Load the trained model + vectorizer from sentiment.pkl
with open("src/sentiment.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
vectorizer = data['vectorizer']

# Streamlit App
st.title("Mama Earth Reviews Sentiment Analysis")
st.write("Analyze sentiment of customer reviews and visualize insights.")

# Input Section
user_input = st.text_area("Enter a review:", "")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]

        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
        st.info(f"Probability: Positive: {prediction_proba[1]:.2f}, Negative: {prediction_proba[0]:.2f}")

        # SHAP Explainability for text
        st.subheader("SHAP Explanation")
        explainer = shap.Explainer(model.predict_proba, X_input)
        shap_values = explainer(X_input)

        # Plot SHAP values
        st.pyplot(shap.plots.text(shap_values[0]))
    else:
        st.warning("Please enter a review to analyze.")
