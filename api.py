import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import string

model = joblib.load("bullying_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.get("/")
def home():
    return {"message": "CyberSafe bullying detection API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    cleaned = clean_text(request.text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = float(max(probabilities))

    return {
        "text": request.text,
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "is_bullying": prediction == "bullying"
    }