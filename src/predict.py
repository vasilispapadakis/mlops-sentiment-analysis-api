from src.model import load_model
from src.logger import logging

model = load_model()

def predict_sentiment(text):
    prediction = model.predict([text])[0]
    prob = model.predict_proba([text])[0].max()
    return {"sentiment": "Negative" if prediction else "Positive", "confidence": prob}

def test_model():
    from sklearn.metrics import accuracy_score
    from src.model import load_data
    movie_reviews_data_folder = 'data/aclImdb/test/'
    dataset = load_data(movie_reviews_data_folder)
    X_test, y_test = dataset.data, dataset.target
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    return accuracy
