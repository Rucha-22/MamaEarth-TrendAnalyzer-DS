import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_preprocess

# Ensure src folder exists
os.makedirs("src", exist_ok=True)

# Load and preprocess dataset
df = load_and_preprocess("data/dataframe_with_category_modified.csv")

X = df['clean_text']
y = df['Sentiment_mapped']  # 0 = negative, 1 = neutral, 2 = positive

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build improved pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,     # larger vocabulary
        ngram_range=(1,3),      # unigrams + bigrams + trigrams
        stop_words='english'    # remove stopwords
    )),
    ("clf", LogisticRegression(
        max_iter=2000, 
        class_weight="balanced", 
        C=1.5,                  # tuned regularization
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    ))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# Save the trained pipeline
model_path = "src/sentiment_pipeline.pkl"
joblib.dump(pipeline, model_path)
print(f"✅ Model pipeline saved at: {model_path}")
