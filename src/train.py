import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Ensure src folder exists
os.makedirs("src", exist_ok=True)

# -------------------------------
# Load dataset directly from CSV
# -------------------------------
df = pd.read_csv("data/dataframe_with_category_modified.csv")
print(df.head())

# Minimal text cleaning: lowercase
df['clean_text'] = df['Review Texts'].str.lower()

# Check class distribution
print("Class distribution:\n", df['Sentiment_mapped'].value_counts())

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['Sentiment_mapped'],
    test_size=0.2, random_state=42, stratify=df['Sentiment_mapped']
)
print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -------------------------------
# Vectorize text
# -------------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF shape (train):", X_train_tfidf.shape)
print("TF-IDF shape (test):", X_test_tfidf.shape)

# -------------------------------
# Train multiple models
# -------------------------------
models = {
    "BernoulliNB": BernoulliNB(),
    "LinearSVC": LinearSVC(max_iter=2000, class_weight='balanced', random_state=42),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight='balanced', C=1.5,
        multi_class='multinomial', solver='lbfgs', random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n", classification_report(
        y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']
    ))
    results[name] = {"model": model, "accuracy": acc}

# -------------------------------
# Select the best model
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

# Save the best model and vectorizer together
joblib.dump({"model": best_model, "vectorizer": vectorizer}, "src/best_sentiment_model.pkl")
print("✅ Best model and vectorizer saved at: src/best_sentiment_model.pkl")
