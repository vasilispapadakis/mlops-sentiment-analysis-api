from fastapi import FastAPI
from src.predict import predict_sentiment

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
def predict(text: str):
    result = predict_sentiment(text)
    return result

