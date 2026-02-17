import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_df["review_text"])
X_val = vectorizer.transform(val_df["review_text"])

y_train = train_df["label"]
y_val = val_df["label"]

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_val)

print("\nClassification Report:")
print(classification_report(y_val, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, preds))

# Save model
joblib.dump(model, "review_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
