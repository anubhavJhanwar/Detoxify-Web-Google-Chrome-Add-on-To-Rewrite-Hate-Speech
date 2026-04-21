# Detoxify Web — Chrome Add-on to Rewrite Hate Speech

An explainable NLP system that detects toxic text, explains why it is toxic, and rewrites it into a neutral form — with a Chrome Extension for real-time usage.

## Features
- Detects toxic text using Logistic Regression  
- Provides explainable predictions (feature contributions)  
- Rewrites toxic sentences into neutral form  
- Uses only classical NLP (spaCy + NLTK)  
- Chrome Extension for real-time text modification  

## Tech Stack
- Backend: FastAPI, Python 3.11  
- ML: Scikit-learn (Logistic Regression)  
- NLP: spaCy, NLTK (VADER), TF-IDF  
- Frontend: HTML, CSS, JavaScript  
- Extension: Chrome Manifest V3  

## Project Structure
backend/  
frontend/  
extension/  
data/  

## API Endpoints
- POST /analyze — full pipeline  
- POST /preprocess — preprocessing only  
- POST /train — retrain model  
- POST /rewrite — rewrite text  
- GET /health — health check  

## How to Run
1. python setup.py  
2. cd backend  
3. python train_model.py ../data/jigsaw_dataset.csv  
4. uvicorn app:app --reload --port 8000  
5. cd ../frontend  
6. python -m http.server 3000  

Load Chrome Extension:  
chrome://extensions → Load unpacked → extension/

## Model Performance
- Accuracy: 89.1%  
- Precision: 91.0%  
- Recall: 86.9%  
- F1 Score: 88.9%  

## License
MIT License