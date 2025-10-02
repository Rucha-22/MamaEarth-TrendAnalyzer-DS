import joblib
from preprocess import clean_text

def test_pipeline_prediction(sample_review):
    """
    Test prediction using the saved full pipeline
    """
    # Load the trained pipeline
    pipeline = joblib.load("src/sentiment_pipeline.pkl")

    # Preprocess the review
    sample_clean = clean_text(sample_review)

    # Predict sentiment
    pred = pipeline.predict([sample_clean])[0]

    # Define valid classes
    valid_classes = [0, 1, 2]
    assert pred in valid_classes, f"Predicted class {pred} not in valid classes {valid_classes}"

    # Map to readable sentiment
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"[Pipeline] Review: {sample_review}")
    print(f"[Pipeline] Predicted Sentiment: {sentiment_mapping[pred]}\n")


def test_separate_prediction(sample_review):
    """
    Test prediction using the separate TF-IDF vectorizer and model
    """
    # Load separate TF-IDF vectorizer and model
    vectorizer = joblib.load("src/tfidf_vectorizer.pkl")
    model = joblib.load("src/sentiment_model.pkl")

    # Preprocess the review
    sample_clean = clean_text(sample_review)

    # Transform text and predict
    X_vec = vectorizer.transform([sample_clean])
    pred = model.predict(X_vec)[0]

    # Define valid classes
    valid_classes = [0, 1, 2]
    assert pred in valid_classes, f"Predicted class {pred} not in valid classes {valid_classes}"

    # Map to readable sentiment
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"[Separate Model] Review: {sample_review}")
    print(f"[Separate Model] Predicted Sentiment: {sentiment_mapping[pred]}\n")


if __name__ == "__main__":
    # Sample review to test
    sample_review = "The Mamaearth product is really great and useful!"

    # Test pipeline prediction
    test_pipeline_prediction(sample_review)

    # Test separate TF-IDF + model prediction
    test_separate_prediction(sample_review)
