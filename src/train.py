import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocess import load_and_preprocess

# Ensure 'src' folder exists
os.makedirs("src", exist_ok=True)

# Load and preprocess dataset
df = load_and_preprocess("data/dataframe_with_category_modified.csv")

# Features and target
X = df['clean_text']
y = df['Sentiment_mapped']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Option 1: Pipeline (TF-IDF + Logistic Regression)
# -----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Pipeline Accuracy:", accuracy_score(y_test, y_pred))
print("\nPipeline Classification Report:\n", classification_report(y_test, y_pred))

# Save the entire pipeline
pipeline_path = "src/sentiment_pipeline.pkl"
joblib.dump(pipeline, pipeline_path)
print(f"✅ Pipeline saved at: {pipeline_path}")

# -----------------------------
# Option 2: Save TF-IDF and model separately
# -----------------------------
# Vectorize separately
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression separately
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate separately
y_pred_sep = model.predict(X_test_vec)
print("Separate Model Accuracy:", accuracy_score(y_test, y_pred_sep))
print("\nSeparate Model Classification Report:\n", classification_report(y_test, y_pred_sep))

# Save separate artifacts
vectorizer_path = "src/tfidf_vectorizer.pkl"
model_path = "src/sentiment_model.pkl"
joblib.dump(vectorizer, vectorizer_path)
joblib.dump(model, model_path)
print(f"✅ TF-IDF vectorizer saved at: {vectorizer_path}")
print(f"✅ Logistic Regression model saved at: {model_path}")
