# IMDB Sentiment Analysis

## Project Overview

This project is a Machine Learning based Sentiment Analysis system for classifying IMDB movie reviews as Positive or Negative.

The project uses Natural Language Processing (NLP) techniques to preprocess movie reviews, convert text into numerical features using TF-IDF, and classify the sentiment using a Multinomial Naive Bayes model.

## Objective

The main objective of this project is to automatically identify whether a given movie review expresses a positive or negative sentiment.

## Dataset

The project uses the IMDB movie review dataset.

The dataset contains movie reviews along with their corresponding sentiment labels:

- Positive
- Negative

The cleaned dataset contains the following columns:

- `review`
- `sentiment`
- `cleaned_review`

## Text Preprocessing

The movie reviews are cleaned and preprocessed before model training.

The preprocessing pipeline includes:

- Text normalization
- Removal of unwanted content
- Tokenization
- Stopword handling
- Lemmatization

The cleaned reviews are stored in `IMDB_Dataset_Cleaned.csv`.

## Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the cleaned text into numerical feature vectors.

The model uses:

- Unigrams
- Bigrams

## Machine Learning Model

The sentiment classifier used in this project is:

**Multinomial Naive Bayes**

The model is trained using the TF-IDF feature vectors and the sentiment labels.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

The initial model evaluation achieved approximately **88% accuracy**.

## Web Application

A Streamlit web application is developed to allow users to enter their own movie reviews.

The application:

1. Accepts a movie review from the user.
2. Converts the review into TF-IDF features.
3. Predicts the sentiment.
4. Displays Positive or Negative sentiment.
5. Displays the model confidence percentage.

## Project Files

```text
IMDB_Sentiment_Project/
│
├── app.py
├── part2_clean.py
├── part3_classify.py
├── IMDB_Dataset_Cleaned.csv
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── README.md