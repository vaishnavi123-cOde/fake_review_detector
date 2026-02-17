from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import os
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)

# Initialize app FIRST
app = FastAPI(
    title="Fake Review Detection API",
    description="Detects whether a product review is fake or genuine using NLP.",
    version="1.0"
)

# Load model and vectorizer
model = joblib.load("review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Serve frontend
@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

class ReviewRequest(BaseModel):
    review_text: str

@app.post("/predict")
def predict_review(request: ReviewRequest):
    text = request.review_text

    vec = vectorizer.transform([text])
    prob = model.predict_proba(vec)[0][1]
    prediction = int(prob > 0.6)

    return {
        "prediction": "FAKE" if prediction == 1 else "REAL",
        "fake_probability": round(float(prob), 4)
    }
