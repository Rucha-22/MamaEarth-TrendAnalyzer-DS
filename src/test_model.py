import joblib
from preprocess import clean_text

def test_prediction():
    # Load the combined pipeline
    pipeline = joblib.load("src/sentiment_pipeline.pkl")

    # Sample review
    sample_review = "The Mamaearth product is really great and useful!"
    sample_clean = clean_text(sample_review)

    # Predict directly using the pipeline
    pred = pipeline.predict([sample_clean])
    pred_proba = pipeline.predict_proba([sample_clean])[0]  # optional probabilities

    # Print results
    print(f"Review: {sample_review}")
    print(f"Cleaned Review: {sample_clean}")
    print(f"Predicted class: {pred[0]}")
    print(f"Prediction probabilities: {pred_proba}")

    # Optional: Check that prediction is a valid class
    assert pred[0] in [0, 1, 2], f"Unexpected class: {pred[0]}"  # adjust classes if needed
    print("Test passed ✅")

# Run the test
if __name__ == "__main__":
    test_prediction()
