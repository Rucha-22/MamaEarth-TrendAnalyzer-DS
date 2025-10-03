import joblib
from preprocess import clean_text

def test_prediction():
    # Load the combined pipeline
    pipeline = joblib.load("src/best_sentiment_model.pkl")

    # Sample review
    sample_review = "The Mamaearth product is really great and useful!"
    sample_clean = clean_text(sample_review)

    # Predict directly using the pipeline
    pred = pipeline['model'].predict(pipeline['vectorizer'].transform([sample_clean]))

    # Map numerical prediction to sentiment label
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_sentiment = sentiment_map.get(pred[0], 'Unknown')

    # Check if predict_proba is available for the model
    if hasattr(pipeline['model'], 'predict_proba'):
      pred_proba = pipeline['model'].predict_proba(pipeline['vectorizer'].transform([sample_clean]))[0]  # optional probabilities
      print(f"Prediction probabilities: {pred_proba}")
    else:
      print("Model does not support predict_proba.")

    # Print results
    print(f"Review: {sample_review}")
    print(f"Cleaned Review: {sample_clean}")
    print(f"Predicted class: {pred[0]} ({predicted_sentiment})")


    # Optional: Check that prediction is a valid class
    assert pred[0] in [0, 1, 2], f"Unexpected class: {pred[0]}"  # adjust classes if needed
    print("Test passed ✅")

# Run the test
if __name__ == "__main__":
    test_prediction()
