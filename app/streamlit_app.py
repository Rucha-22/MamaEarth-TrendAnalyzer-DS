import streamlit as st
import joblib
import os
import numpy as np


def load_pickle(file_path, name="object"):
    if not os.path.exists(file_path):
        st.error(f"❌ {name} file not found: {file_path}")
        return None
    try:
        obj = joblib.load(file_path)
        st.success(f"✅ Loaded {name} from '{file_path}'.")
        return obj
    except Exception as e:
        st.error(f"⚠️ Failed to load {name}: {e}")
        return None


scaler_file = "src/scaler.pkl"
model_file = "src/best_regression_model1.pkl"

scaler = load_pickle(scaler_file, "Scaler")
model = load_pickle(model_file, "Regression Model")

if scaler is None or model is None:
    st.stop()


st.title("⭐ Mama Earth Product Rating Prediction")

st.write("Fill in the product details below to predict its rating (1–5 scale).")


product_name = st.text_input("Product Name", "Face Cream with Vitamin C")
mrp = st.number_input("MRP (₹)", min_value=0.0, step=10.0, value=599.0)

sentiment = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])

category = st.radio(
    "Category",
    ["Face", "Hair", "Other"],
    horizontal=True
)

# One-hot encode category
cat_face = 1 if category == "Face" else 0
cat_hair = 1 if category == "Hair" else 0
cat_other = 1 if category == "Other" else 0


if st.button("Predict Rating"):
    try:
        # Encode sentiment
        sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
        sentiment_val = sentiment_map.get(sentiment, 0)

        # Create feature array
        X_input = np.array([[mrp, sentiment_val, cat_face, cat_hair, cat_other]])

        # Auto-adjust to scaler feature count
        expected = getattr(scaler, "n_features_in_", X_input.shape[1])
        if X_input.shape[1] < expected:
            pad = np.zeros((1, expected - X_input.shape[1]))
            X_input = np.hstack([X_input, pad])
        elif X_input.shape[1] > expected:
            X_input = X_input[:, :expected]

        # Scale and predict
        X_scaled = scaler.transform(X_input)
        predicted_rating = model.predict(X_scaled)[0]

        # Clip rating to 1–5 range
        predicted_rating = float(np.clip(predicted_rating, 1, 5))

        st.success(f"⭐ Predicted Product Rating: {predicted_rating:.2f} / 5")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
