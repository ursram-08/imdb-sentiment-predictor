from copyreg import pickle

import pandas as pd

df = pd.read_csv("IMDB_Dataset_Cleaned.csv")

print("Dataset loaded successfully")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head())
import pandas as pd

# Load the cleaned dataset generated in Part 2
df = pd.read_csv("IMDB_Dataset_Cleaned.csv")

print("Original rows:", len(df))

# For initial testing, use only 20,000 rows
df = df.iloc[:20000].copy()

print("Rows used for testing:", len(df))
print(df.head())
df = df.iloc[:20000].copy()
from sklearn.model_selection import train_test_split

X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    max_features=50000,
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Baseline Accuracy: 0.500")
print("Improvement:", accuracy - 0.500)
from sklearn.metrics import classification_report
print("\nClassification Report")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)
import numpy as np
feature_names = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_
model.classes_
negative_index = list(model.classes_).index("negative")
positive_index = list(model.classes_).index("positive")
print("\nTop 20 Positive Terms:")
top_positive = np.argsort(
    log_probs[positive_index] - log_probs[negative_index]
)[::-1][:20]

top_negative = np.argsort(
    log_probs[negative_index] - log_probs[positive_index]
)[::-1][:20]
print("\nTop 20 Positive Terms:")

for i in top_positive:
    print(feature_names[i])

print("\nTop 20 Negative Terms:")

for i in top_negative:
    print(feature_names[i])
    # Step 15: Manual Review Prediction

review = input("\nEnter a movie review: ")

review_tfidf = vectorizer.transform([review])

prediction = model.predict(review_tfidf)[0]

if prediction == "positive":
    print("Prediction: POSITIVE 🟢")
else:
    print("Prediction: NEGATIVE 🔴")# Save trained model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and Vectorizer saved successfully!")