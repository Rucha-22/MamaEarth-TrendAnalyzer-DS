from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from preprocess import load_and_preprocess

# Ensure src folder exists
os.makedirs("src", exist_ok=True)

# Load and preprocess dataset
df = load_and_preprocess("data/dataframe_with_category_modified.csv")

X = df['clean_text']
y = df['Sentiment_mapped']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build Pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save pipeline (single .pkl)
model_path = "src/sentiment_pipeline.pkl"
joblib.dump(pipeline, model_path)

print(f"✅ Model pipeline saved at: {model_path}")
