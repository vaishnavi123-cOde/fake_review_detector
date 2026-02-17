Fake Review Detection System

Live Demo:
👉 https://fake-review-detector-kcqw.onrender.com

📌 Overview

This project is an end-to-end NLP system that detects whether a product review is genuine (REAL) or computer-generated / fake (FAKE).

It includes:

Data preprocessing pipeline

Machine learning model training

Evaluation metrics

FastAPI backend

Interactive frontend

Public cloud deployment (Render)

🧠 Problem Statement

Online platforms are flooded with fake or AI-generated reviews that mislead users.

This system analyzes review text and predicts the likelihood that a review is fake based on learned linguistic patterns.

📊 Dataset

~40,000 labeled reviews

Balanced classes:

CG (Computer Generated / Fake)

OR (Original / Real)

After cleaning:

40,409 usable reviews

Stratified train/validation/test split

 Approach
1️⃣ Preprocessing

Removed duplicates

Removed very short reviews

Converted labels:

OR → 0

CG → 1

2️⃣ Feature Engineering

TF-IDF vectorization

1-gram and 2-gram features

10,000 max features

3️⃣ Model

Logistic Regression

Threshold adjusted to 0.6 to reduce false positives

📈 Model Performance

Validation Results:

Accuracy: 0.88

F1 Score: 0.88

Precision: 0.89

Recall: 0.87

Confusion Matrix:

[[2704  328]
 [ 384 2645]]

 System Architecture

Frontend → FastAPI Backend → TF-IDF Vectorizer → Logistic Regression Model → Prediction Response

 Deployment

Backend: FastAPI

Hosting: Render (Free Tier)

Public API endpoint

Interactive UI with confidence visualization

 Tech Stack

Python

scikit-learn

FastAPI

Uvicorn

HTML/CSS/JavaScript

Render (Cloud Hosting)

Git & GitHub

 Example

Input:

This product is absolutely amazing amazing amazing!!! Highly recommended!!!


Output:

Prediction: FAKE
Fake Probability: 0.91

Limitations

Model detects stylistic patterns, not factual truth.

Gibberish input may sometimes be labeled REAL due to absence of fake signals.

Does not verify user identity or purchase authenticity.

 Future Improvements

Add SHAP explainability

Compare with LinearSVC / XGBoost

Upgrade to Sentence-BERT

Add evaluation dashboard page

Add rating + review length features



Vaishnavi Ganti
B.Tech CSE | Machine Learning & AI
