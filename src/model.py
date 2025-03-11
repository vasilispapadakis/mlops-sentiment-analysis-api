import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

def load_data(container_path):
    categories = ['pos', 'neg']
    data = Bunch()
    data.data = []
    data.target = []
    data.target_names = categories
    for label, category in enumerate(categories):
        path = os.path.join(container_path, category)
        for file_name in sorted(os.listdir(path)):
            with open(os.path.join(path, file_name), 'r', encoding='utf-8') as f:
                data.data.append(f.read())
                data.target.append(label)
    return data

def train_and_save_model():
    # Load training data
    movie_reviews_data_folder = 'data/aclImdb/train/'
    dataset = load_data(movie_reviews_data_folder)
    X_train, y_train = dataset.data, dataset.target
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    # Save model pipeline
    joblib.dump(pipeline, 'artifacts/sentiment_model.pkl')

def load_model(model_path='artifacts/sentiment_model.pkl'):
    return joblib.load(model_path)


