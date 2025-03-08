# Sentiment Analysis API with FastAPI, Docker, and MLflow

This project aims to build a REST API for classifying movie reviews as positive or negative using machine learning. The API is built using FastAPI and will be deployed in a Docker container for ease of use in various environments. 

## Project Structure

The project is organized as follows:

```
mlops-sentiment-analysis-api
├── data            # Dataset folder (contains IMDb dataset)
├── src             # Source code
│   ├── logger.py   # Logging configuration (to be implemented)
│   ├── main.py     # FastAPI API definition (basic skeleton)
│   ├── model.py    # Placeholder for ML model pipeline
│   └── predict.py  # Placeholder for prediction logic
├── tests           # Tests folder (to be developed)
├── Dockerfile      # Docker specification file (to be developed)
├── requirements.txt# Python package dependencies (to be developed)
└── README.md       # Project information and setup guide
```