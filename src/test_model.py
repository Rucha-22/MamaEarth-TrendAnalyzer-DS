import joblib
from preprocess import clean_text

def test_prediction():
    model = joblib.load("mamaearth_sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    sample_review = "The Mamaearth product is really great and useful!"
    sample_clean = clean_text(sample_review)
    sample_vec = vectorizer.transform([sample_clean])

    pred = model.predict(sample_vec)
    assert pred[0] in [0, 1, 2]  # check valid sentiment classes
