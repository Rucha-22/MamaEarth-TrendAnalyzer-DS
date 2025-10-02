import joblib
from preprocess import clean_text

def test_prediction():
    # Load the combined pipeline
    pipeline = joblib.load("src/sentiment_pipeline.pkl")

    sample_review = "The Mamaearth product is really great and useful!"
    sample_clean = clean_text(sample_review)

    # Predict directly using the pipeline
    pred = pipeline.predict([sample_clean])

    # Check that prediction is a valid sentiment class
    assert pred[0] in [0, 1, 2]  # adjust classes if your mapping is different
