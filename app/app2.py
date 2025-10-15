import streamlit as st
import joblib
import os
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import pandas as pd

# ---------- CONFIG ----------
MODEL_PATH = "src/best_regression_model.pkl"

# ---------- Utility: Load Model Bundle Safely ----------
def load_model_bundle(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}")
        return None
    try:
        obj = joblib.load(path)
        # Case 1: Full bundle dict (as in ds_exp5.py)
        if isinstance(obj, dict):
            st.success(f"✅ Loaded model bundle: {obj.get('model_name', 'Unknown')}")
            return obj
        # Case 2: Just a model (no dict)
        else:
            st.warning("⚠️ Model file contains only the estimator (no scaler/features). Using defaults.")
            return {
                "model_name": type(obj).__name__,
                "estimator": obj,
                "scaler": None,
                "features": ["MRP", "name_keyword_score", "Sentiment_encoded", "Category_encoded"]
            }
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None


# ---------- Load Model ----------
bundle = load_model_bundle(MODEL_PATH)
if bundle is None:
    st.stop()

model = bundle["estimator"]
scaler = bundle["scaler"]
feature_names = bundle["features"]

# ---------- Streamlit UI ----------
st.title("⭐ Mama Earth Product Rating Prediction")
st.write("Fill in the product details below to predict its rating (1–5 scale).")

product_name = st.text_input("🧴 Product Name", "Face Cream with Vitamin C")
mrp = st.number_input("💰 MRP (₹)", min_value=0.0, step=10.0, value=599.0)
sentiment = st.selectbox("🗣️ Sentiment", ["Positive", "Neutral", "Negative"])
category = st.radio("📦 Category", ["Face", "Hair", "Other"], horizontal=True)

# ---------- Feature Engineering (match ds_exp5.py) ----------
sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
sentiment_val = sentiment_map[sentiment]

cat_map = {"Face": 0, "Hair": 1, "Other": 2}
category_val = cat_map[category]

# Approximate "name_keyword_score" as normalized length
name_keyword_score = len(product_name) / 100.0
name_keyword_score = np.clip(name_keyword_score, 0, 1)

# Input vector aligned with training features
X_input = np.array([[mrp, name_keyword_score, sentiment_val, category_val]])

# ---------- Predict Button ----------
if st.button("🎯 Predict Rating"):
    try:
        # Apply scaling if available
        X_scaled = scaler.transform(X_input) if scaler is not None else X_input

        prediction = model.predict(X_scaled)[0]
        predicted_rating = float(np.clip(prediction, 1, 5))

        st.success(f"⭐ Predicted Product Rating: **{predicted_rating:.2f} / 5**")

        # Display input features
        st.subheader("🔢 Input Features")
        st.dataframe(pd.DataFrame(X_input, columns=feature_names))

        # ---------- SHAP Explanation ----------
        st.subheader("🔍 SHAP Explanation")
        try:
            explainer = shap.Explainer(model, X_scaled)
            shap_values = explainer(X_scaled)

            fig, ax = plt.subplots()
            shap.waterfall_plot(shap_values[0], max_display=5, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")

        # ---------- LIME Explanation ----------
        st.subheader("🧠 LIME Explanation")
        try:
            # Generate synthetic background around the input
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 0.1, size=(100, X_input.shape[1]))  # small random noise
            background = np.clip(
                X_input + noise * np.maximum(X_input.std(axis=1, keepdims=True), 1),
                0,
                None,
            )

            lime_explainer = LimeTabularExplainer(
                training_data=background,
                feature_names=feature_names,
                mode="regression"
            )

            # Predict function for LIME
            def predict_fn(x):
                x_scaled = scaler.transform(x) if scaler else x
                preds = model.predict(x_scaled)
                return preds.flatten()  # ensure 1D

            exp = lime_explainer.explain_instance(
                data_row=X_input[0],
                predict_fn=predict_fn,
                num_features=len(feature_names)
            )

            st.pyplot(exp.as_pyplot_figure())
        except Exception as e:
            st.warning(f"LIME explanation failed: {e}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
