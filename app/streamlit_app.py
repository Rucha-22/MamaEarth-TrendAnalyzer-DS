import streamlit as st
import joblib
import os
import numpy as np

# -----------------------------
# Safe file loader
# -----------------------------
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

# -----------------------------
# Load Scaler and Model
# -----------------------------
scaler_file = "src/scaler.pkl"
model_file = "src/best_regression_model.pkl"

scaler = load_pickle(scaler_file, "Scaler")
model = load_pickle(model_file, "Regression Model")

if scaler is None or model is None:
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("⭐ Mama Earth Product Rating Prediction")

st.write("Enter a product review below to predict its rating (1–5 scale).")

user_input = st.text_area("Enter your review:")

if st.button("Predict Rating"):

    if not user_input.strip():
        st.warning("Please enter a review.")
    else:
        try:
            # -----------------------------
            # Create dummy numeric features
            # -----------------------------
            # You can modify this later when you know your real feature set
            review = user_input.strip()
            f1 = len(review)                                  # total characters
            f2 = len(review.split())                          # total words
            f3 = review.count('!')                            # number of exclamations
            f4 = sum(c.isupper() for c in review)             # uppercase count
            X_input = np.array([[f1, f2, f3, f4]])

            # -----------------------------
            # Auto-adjust to scaler feature count
            # -----------------------------
            expected = getattr(scaler, "n_features_in_", X_input.shape[1])
            if X_input.shape[1] < expected:
                pad = np.zeros((1, expected - X_input.shape[1]))
                X_input = np.hstack([X_input, pad])
            elif X_input.shape[1] > expected:
                X_input = X_input[:, :expected]

            # -----------------------------
            # Scale and predict
            # -----------------------------
            X_scaled = scaler.transform(X_input)
            predicted_rating = model.predict(X_scaled)[0]

            # Clip to rating range
            predicted_rating = float(np.clip(predicted_rating, 1, 5))

            st.success(f"⭐ Predicted Product Rating: {predicted_rating:.2f} / 5")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
